"""MFLM Phase 3: WikiText-2 benchmark — MFLM vs Baseline.

Trains both models side by side, evaluates on validation set,
prints comparison table.
"""
import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from mflm.model import MFLM, BaselineLM, MFLMConfig
from mflm.data import get_dataloader


def get_lr(it, lr_max, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return lr_max * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr_max - min_lr)


SEQ_LEN = 256
VOCAB_SIZE = 50257


def train_model(model_name, model, args):
    print(f"\n{'='*60}", flush=True)
    print(f"  Training {model_name}  |  params: {model.n_params():,}", flush=True)
    print(f"{'='*60}", flush=True)

    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.98)
    )
    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'wikitext2')
    train_dl = get_dataloader(os.path.join(data_dir, 'train.bin'), SEQ_LEN, args.batch, shuffle=True)
    valid_dl = get_dataloader(os.path.join(data_dir, 'valid.bin'), SEQ_LEN, args.batch, shuffle=False)

    best_val_loss = float('inf')
    best_val_ppl = float('inf')

    model.train()
    step = 0
    t0 = time.time()

    warmup_iters = 1000
    lr_max = 3e-4
    min_lr = 3e-5

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_best.pt')

    train_iter = iter(train_dl)

    while step < args.steps:
        lr = get_lr(step, lr_max, warmup_iters, args.steps, min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / args.accum
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.accum
                loss.backward()

            accum_loss += loss.item()

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1

        if step % 100 == 0:
            t1 = time.time()
            dt = t1 - t0
            tok_s = (args.batch * SEQ_LEN * args.accum * 100) / dt
            print(f"  [{model_name}] step {step:5d}/{args.steps} | "
                  f"loss {accum_loss:.4f} | lr {lr:.2e} | "
                  f"{tok_s:,.0f} tok/s", flush=True)
            t0 = time.time()

        if step % 500 == 0:
            val_loss = evaluate(model, valid_dl, device, use_amp)
            val_ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow
            print(f"  [{model_name}] step {step:5d} | "
                  f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}", flush=True)

            if hasattr(model, 'diagnostics'):
                diag = model.diagnostics()
                if diag:
                    d = diag[-1]  # last iteration step
                    print(f"  [{model_name}] mass: mean={d['mass_mean']:.3f} "
                          f"std={d['mass_std']:.3f} "
                          f"+={d['frac_positive']:.2f} -={d['frac_negative']:.2f} "
                          f"L={d['frac_lagrange']:.2f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), ckpt_path)
            model.train()

    return {
        'params': model.n_params(),
        'best_val_loss': best_val_loss,
        'best_val_ppl': best_val_ppl,
    }


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=30):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="MFLM Phase 3: WikiText-2")
    parser.add_argument('--size', default='small', choices=['small', 'medium'])
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.size == 'small':
        cfg = MFLMConfig(
            vocab_size=VOCAB_SIZE, d_model=256, n_heads=4,
            max_steps=6, n_layers=4, d_ff=1024,
            field_window=128, max_seq_len=SEQ_LEN,
        )
    else:
        cfg = MFLMConfig(
            vocab_size=VOCAB_SIZE, d_model=384, n_heads=6,
            max_steps=8, n_layers=6, d_ff=1536,
            field_window=256, max_seq_len=SEQ_LEN,
        )

    print(f"Config: size={args.size}, steps={args.steps}, seed={args.seed}, "
          f"batch={args.batch}x{args.accum}, device={args.device}", flush=True)

    # Train baseline first
    torch.manual_seed(args.seed)
    baseline = BaselineLM(cfg)
    res_base = train_model('Baseline', baseline, args)
    del baseline
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # Then MFLM
    torch.manual_seed(args.seed)
    mflm = MFLM(cfg)
    res_mflm = train_model('MFLM', mflm, args)
    del mflm
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # Summary
    delta = (res_mflm['best_val_ppl'] - res_base['best_val_ppl']) / res_base['best_val_ppl'] * 100
    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS: WikiText-2, size={args.size}, seed={args.seed}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Model':<12} | {'Params':>10} | {'Val PPL':>10}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'Baseline':<12} | {res_base['params']:>10,} | {res_base['best_val_ppl']:>10.2f}", flush=True)
    print(f"  {'MFLM':<12} | {res_mflm['params']:>10,} | {res_mflm['best_val_ppl']:>10.2f}", flush=True)
    print(f"  MFLM delta: {delta:+.2f}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
