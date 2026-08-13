"""Training script for MFLM Exoskeleton."""

import os
import time
import torch
from transformers import AutoTokenizer

from mflm.model import MFLMConfig
from mflm.exoskeleton import MFLMExoskeleton

def get_text_batches(txt_path: str, tokenizer, batch_size: int, seq_len: int):
    """Yield batches of input_ids directly from the raw text using Qwen's tokenizer."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    stories = text.split('<|endoftext|>')
    all_tokens = []
    
    print("Tokenizing data with Qwen tokenizer...")
    # Just grab enough stories to get a few thousand tokens for a quick test
    for story in stories[:100]:
        if not story.strip(): continue
        tokens = tokenizer.encode(story)
        all_tokens.extend(tokens)
        if len(all_tokens) > batch_size * seq_len * 100:  # 100 batches is enough
            break
            
    # Convert to tensor
    data = torch.tensor(all_tokens, dtype=torch.long)
    
    # Yield batches
    idx = 0
    while idx + batch_size * seq_len + 1 <= len(data):
        batch = data[idx : idx + batch_size * seq_len + 1]
        x = batch[:-1].view(batch_size, seq_len)
        y = batch[1:].view(batch_size, seq_len)
        yield x, y
        idx += batch_size * seq_len

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading tokenizer {qwen_id}...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_id)
    
    # Qwen1.5 vocab is 151936
    mflm_cfg = MFLMConfig(
        vocab_size=151936,  # Must match Qwen
        d_model=128,        # Tiny scout
        d_ff=512,
        n_heads=4,
        max_steps=4,
        field_window=128,
        max_seq_len=256
    )
    
    model = MFLMExoskeleton(mflm_cfg, qwen_path=qwen_id)
    model.to(device)
    
    # Print params
    trainable = model.n_trainable_params()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({(trainable/total)*100:.2f}%)")
    
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01
    )
    
    txt_path = "bench/data/tinystories/TinyStoriesV2-GPT4-valid.txt"
    if not os.path.exists(txt_path):
        print(f"File {txt_path} not found. Please run prepare_tinystories.py first.")
        return
        
    batches = list(get_text_batches(txt_path, tokenizer, batch_size=4, seq_len=128))
    print(f"Prepared {len(batches)} batches.")
    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    print("Starting test training loop...")
    for step, (x, y) in enumerate(batches[:50]):
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = model(x, labels=y)
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        print(f"Step {step:2d} | loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
