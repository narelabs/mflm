import matplotlib.pyplot as plt
import numpy as np
import os

def create_plots():
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Long Context Scaling: Time
    seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    base_time = [10.9, 21.4, 45.2, 116.6, 301.4, 921.3]
    mflm_time = [14.6, 26.7, 57.4, 138.4, 263.0, 517.8]
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(seq_lens, base_time, 'o-', color='#e74c3c', linewidth=2.5, label='Baseline (Transformer) - $O(N^2)$')
    plt.plot(seq_lens, mflm_time, 's-', color='#2ecc71', linewidth=2.5, label='MFLM (Mass-Field) - $O(N)$')
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(seq_lens, labels=[str(s) for s in seq_lens])
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.title('Inference Time vs Sequence Length', fontsize=14, fontweight='bold')
    plt.xlabel('Sequence Length (Tokens)', fontsize=12)
    plt.ylabel('Forward Pass Time (ms)', fontsize=12)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'scaling_time.png'))
    plt.close()

    # 2. Long Context Scaling: VRAM
    base_vram = [122.2, 172.1, 275.5, 489.8, 949.2, 1985.8]
    mflm_vram = [112.4, 161.1, 260.3, 458.6, 856.1, 1648.7]
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(seq_lens, base_vram, 'o-', color='#e74c3c', linewidth=2.5, label='Baseline (Transformer)')
    plt.plot(seq_lens, mflm_vram, 's-', color='#2ecc71', linewidth=2.5, label='MFLM')
    
    plt.xscale('log', base=2)
    plt.xticks(seq_lens, labels=[str(s) for s in seq_lens])
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.title('Peak VRAM Allocation vs Sequence Length', fontsize=14, fontweight='bold')
    plt.xlabel('Sequence Length (Tokens)', fontsize=12)
    plt.ylabel('VRAM (MB)', fontsize=12)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'scaling_vram.png'))
    plt.close()
    
    # 3. Efficiency Comparison Bar Chart
    labels = ['Baseline (Transformer)', 'MFLM']
    params = [16.0, 13.6]
    ppl = [530.5, 559.2]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=150)
    
    color1 = '#3498db'
    ax1.set_ylabel('Parameters (Millions)', color=color1, fontsize=12, fontweight='bold')
    ax1.bar(x - width/2, params, width, color=color1, label='Params (M)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 20)
    
    ax2 = ax1.twinx()
    color2 = '#9b59b6'
    ax2.set_ylabel('Perplexity (Lower is better)', color=color2, fontsize=12, fontweight='bold')
    ax2.bar(x + width/2, ppl, width, color=color2, label='Val PPL')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 700)
    
    # Add values on top of bars
    for i, v in enumerate(params):
        ax1.text(i - width/2, v + 0.5, f'{v}M', ha='center', va='bottom', fontweight='bold', color=color1)
    for i, v in enumerate(ppl):
        ax2.text(i + width/2, v + 20, f'{v}', ha='center', va='bottom', fontweight='bold', color=color2)
        
    plt.title('WikiText-2 Benchmark (5000 steps)', fontsize=14, fontweight='bold')
    plt.xticks(x, labels, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'efficiency_bar.png'))
    plt.close()

    print(f"Saved academic plots to {assets_dir}")

if __name__ == '__main__':
    create_plots()
