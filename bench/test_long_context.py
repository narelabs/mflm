import time
import torch
from mflm.model import MFLM, BaselineLM, MFLMConfig

def measure_memory_and_speed(model, seq_len, device='cuda', batch=1, n_iters=10):
    model.eval()
    x = torch.randint(0, 50257, (batch, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iters
    
    mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return dt * 1000, mem_mb  # returns ms, MB

def main():
    print("=== Long Context Benchmark ===", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---", flush=True)
        
        # We need to set max_seq_len to the tested seq_len
        # and field_window can scale linearly or stay fixed. We will set field_window=seq_len//2
        cfg = MFLMConfig(
            vocab_size=50257, d_model=256, n_heads=4,
            max_steps=6, n_layers=4, d_ff=1024,
            field_window=min(seq_len // 2, 1024), max_seq_len=seq_len
        )
        
        try:
            baseline = BaselineLM(cfg).to(device)
            base_time, base_mem = measure_memory_and_speed(baseline, seq_len, device)
            del baseline
            torch.cuda.empty_cache()
            print(f"Baseline | Time: {base_time:6.1f} ms | VRAM: {base_mem:6.1f} MB", flush=True)
        except Exception as e:
            print(f"Baseline | ERROR (OOM?)", flush=True)
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
        
        try:
            mflm = MFLM(cfg).to(device)
            mflm_time, mflm_mem = measure_memory_and_speed(mflm, seq_len, device)
            del mflm
            torch.cuda.empty_cache()
            print(f"MFLM     | Time: {mflm_time:6.1f} ms | VRAM: {mflm_mem:6.1f} MB", flush=True)
        except Exception as e:
            print(f"MFLM     | ERROR (OOM?)", flush=True)
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
