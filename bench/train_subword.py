"""Subword training script for MFLM on TinyStories."""

import os
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from mflm.model import MFLMConfig, MFLM
from mflm.data import get_dataloader

@dataclass
class SizeSpec:
    name: str
    d_model: int
    n_heads: int
    max_steps: int  # MFLM iterations
    d_ff: int
    field_window: int

SIZES = {
    # Tiny: ~12M parameters total (10M from 50k vocab embeddings alone)
    "tiny": SizeSpec("tiny", d_model=192, n_heads=6, max_steps=6, d_ff=768, field_window=128),
    
    # Small: ~25M parameters total (19M from embeddings)
    "small": SizeSpec("small", d_model=384, n_heads=6, max_steps=8, d_ff=1536, field_window=256),
    
    # Base: ~60M parameters total
    "base": SizeSpec("base", d_model=768, n_heads=12, max_steps=12, d_ff=3072, field_window=512),
}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="tiny", choices=SIZES.keys())
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    # Data
    # For subwords, vocab is standard GPT-2 size. tiktoken cl100k is bigger, but we used gpt2.
    VOCAB_SIZE = 50257
    SEQ_LEN = 256
    
    valid_bin = Path("bench/data/tinystories/TinyStoriesV2-GPT4-valid.bin")
    if not valid_bin.exists():
        raise FileNotFoundError(f"Missing {valid_bin}. Run prepare_tinystories.py first.")
    
    # We will just train on the validation set for the prototype to verify convergence.
    # In a full run, we would use train.bin for training and valid.bin for eval.
    train_loader = get_dataloader(str(valid_bin), SEQ_LEN, args.batch, shuffle=True)
    
    # Model
    spec = SIZES[args.size]
    cfg = MFLMConfig(
        vocab_size=VOCAB_SIZE,
        d_model=spec.d_model,
        d_ff=spec.d_ff,
        n_heads=spec.n_heads,
        max_steps=spec.max_steps,
        field_window=spec.field_window,
        max_seq_len=SEQ_LEN
    )
    
    model = MFLM(cfg).to(args.device)
    print(f"MFLM size={args.size}, params={model.n_params():,}")
    
    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    # Cosine decay down to 10% of lr
    scheduler = CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda"))
    
    model.train()
    step = 0
    running_loss = 0.0
    
    t0 = time.time()
    
    # We loop over the dataset indefinitely until we reach args.steps
    data_iter = iter(train_loader)
    
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(args.accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
                
            x, y = x.to(args.device), y.to(args.device)
            
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.accum
                
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        running_loss += accum_loss
        step += 1
        
        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            dt = time.time() - t0
            tok_sec = (args.batch * args.accum * SEQ_LEN * args.log_interval) / dt
            print(f"step {step:4d} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {tok_sec:,.0f} tok/s")
            running_loss = 0.0
            t0 = time.time()

if __name__ == "__main__":
    train()
