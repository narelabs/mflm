"""MFLM vs Baseline — head-to-head probe.

Trains both models on Tiny Shakespeare (char-level) and compares:
  - eval perplexity
  - parameter count
  - inference speed
  - MFLM mass diagnostics per iteration step

Protocol (following O-12 from archive):
  - Multiple seeds per config
  - Multiple sizes
  - Report mean ± std
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mflm.model import MFLM, BaselineLM, MFLMConfig


# -----------------------------------------------------------------------
# Size specs
# -----------------------------------------------------------------------

@dataclass
class SizeSpec:
    name: str
    d_model: int
    n_heads: int
    n_layers: int  # baseline layers
    max_steps: int  # MFLM iterations
    d_ff: int
    field_window: int


SIZES = {
    "tiny": SizeSpec("tiny", d_model=128, n_heads=4, n_layers=2,
                     max_steps=4, d_ff=512, field_window=64),
    "small": SizeSpec("small", d_model=256, n_heads=4, n_layers=4,
                      max_steps=6, d_ff=512, field_window=64),
    "medium": SizeSpec("medium", d_model=384, n_heads=6, n_layers=6,
                       max_steps=8, d_ff=768, field_window=128),
}


# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def load_corpus(cache: Path) -> str:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"[probe] downloading tiny shakespeare -> {cache}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, cache)
    return cache.read_text(encoding="utf-8")


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int,
                 n_examples: int, seed: int) -> None:
        self.data = data
        self.seq_len = seq_len
        gen = torch.Generator().manual_seed(seed)
        max_start = data.shape[0] - seq_len - 1
        self.starts = torch.randint(0, max_start, (n_examples,),
                                    generator=gen).tolist()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        return self.data[s:s + self.seq_len], self.data[s + 1:s + 1 + self.seq_len]


def _infinite(loader: DataLoader):
    while True:
        yield from loader


# -----------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------

def train_model(
    model: Any,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    n_steps: int,
    lr: float,
    device: str,
    label: str,
    log_every: int = 200,
) -> dict[str, Any]:
    """Train and evaluate a model, return metrics dict."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    it = _infinite(train_loader)
    t0 = time.perf_counter()
    last_loss = float("nan")

    for step in range(1, n_steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
        if step % log_every == 0 or step == n_steps:
            print(f"  [{label}] step {step:5d}  loss {last_loss:.4f}")

    train_time = time.perf_counter() - t0
    eval_loss = evaluate(model, eval_loader, device)
    inf_ms = bench_inference(model, eval_loader, device)

    result: dict[str, Any] = {
        "label": label,
        "n_params": model.n_params(),
        "train_seconds": round(train_time, 1),
        "final_train_loss": last_loss,
        "eval_loss": eval_loss,
        "eval_ppl": float(torch.tensor(eval_loss).exp().item()),
        "inference_ms_per_seq": round(inf_ms, 3),
    }

    if hasattr(model, "diagnostics"):
        diag = model.diagnostics()
        result["diagnostics"] = diag
        if diag:
            print(f"  [{label}] mass diagnostics:")
            for d in diag:
                print(f"    step {d['step']}: mean={d['mass_mean']:.3f} "
                      f"std={d['mass_std']:.3f} "
                      f"range=[{d['mass_min']:.3f}, {d['mass_max']:.3f}]")

    return result


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def bench_inference(model: Any, loader: DataLoader, device: str,
                    n_warmup: int = 3, n_iter: int = 20) -> float:
    model.eval()
    batch = next(iter(loader))[0].to(device)
    for _ in range(n_warmup):
        model(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.train()
    return elapsed / (n_iter * batch.shape[0]) * 1000.0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_single(
    size_name: str,
    seed: int,
    *,
    steps: int,
    lr: float,
    batch: int,
    seq_len: int,
    device: str,
    corpus_path: str,
) -> dict[str, Any]:
    """Run one baseline + MFLM training pair at given size and seed."""
    print(f"\n{'='*70}")
    print(f"  SIZE={size_name}  SEED={seed}")
    print(f"{'='*70}")

    spec = SIZES[size_name]

    # Load data
    text = load_corpus(Path(corpus_path))
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(data.shape[0] * 0.9)
    train_data, eval_data = data[:split], data[split:]

    train_ds = CharDataset(train_data, seq_len=seq_len, n_examples=4096,
                           seed=seed)
    eval_ds = CharDataset(eval_data, seq_len=seq_len, n_examples=512,
                          seed=seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch, shuffle=False)

    cfg = MFLMConfig(
        vocab_size=len(chars),
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        max_steps=spec.max_steps,
        d_ff=spec.d_ff,
        field_window=spec.field_window,
        max_seq_len=seq_len,
    )

    results = {"size": size_name, "seed": seed}

    # --- Baseline ---
    print(f"\n  TRAIN — baseline (layers={cfg.n_layers})")
    torch.manual_seed(seed)
    baseline = BaselineLM(cfg)
    print(f"  params: {baseline.n_params():,}")
    stat = train_model(
        baseline, train_loader, eval_loader,
        n_steps=steps, lr=lr, device=device,
        label=f"baseline/{size_name}/s{seed}",
    )
    stat["variant"] = "baseline"
    results["baseline"] = stat
    baseline_ppl = stat["eval_ppl"]
    print(f"  -> ppl={baseline_ppl:.2f}  params={stat['n_params']:,}")
    del baseline
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- MFLM ---
    print(f"\n  TRAIN — MFLM (steps={cfg.max_steps}, window={cfg.field_window})")
    torch.manual_seed(seed)
    mflm = MFLM(cfg)
    print(f"  params: {mflm.n_params():,}")
    stat = train_model(
        mflm, train_loader, eval_loader,
        n_steps=steps, lr=lr, device=device,
        label=f"mflm/{size_name}/s{seed}",
    )
    stat["variant"] = "mflm"
    results["mflm"] = stat
    mflm_ppl = stat["eval_ppl"]
    delta = (mflm_ppl - baseline_ppl) / max(baseline_ppl, 1e-9) * 100
    results["delta_ppl_pct"] = round(delta, 2)
    print(f"  -> ppl={mflm_ppl:.2f}  params={stat['n_params']:,}  "
          f"delta={delta:+.1f}%")
    del mflm
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MFLM vs Baseline probe")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sizes", default="tiny",
                        help="Comma-separated: tiny,small,medium")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds per size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus", default="bench/data/shakespeare.txt")
    parser.add_argument("--out", default="bench/results/probe.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [s.strip() for s in args.sizes.split(",")]
    print(f"[probe] device={device}, sizes={sizes}, seeds={args.seeds}, "
          f"steps={args.steps}")

    all_results: list[dict[str, Any]] = []

    for size_name in sizes:
        for seed in range(args.seeds):
            result = run_single(
                size_name, seed,
                steps=args.steps, lr=args.lr, batch=args.batch,
                seq_len=args.seq_len, device=device,
                corpus_path=args.corpus,
            )
            all_results.append(result)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['size']:>8s} seed={r['seed']}  "
              f"baseline={r['baseline']['eval_ppl']:7.2f}  "
              f"mflm={r['mflm']['eval_ppl']:7.2f}  "
              f"delta={r['delta_ppl_pct']:+.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2,
                  default=str)
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
