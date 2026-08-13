"""Download and tokenize a small portion of TinyStories."""

import os
import requests
import tiktoken
import numpy as np
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path("bench/data/tinystories")

# TinyStories URLs
URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
]

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"File {dest_path} already exists, skipping download.")
        return
    
    # Check if we should skip train.txt for now since it's 4.8GB. We can just download valid.txt for a quick test.
    if "train" in url and os.environ.get("SKIP_LARGE_TRAIN"):
        print(f"Skipping {url} as SKIP_LARGE_TRAIN is set.")
        return

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = f.write(data)
            bar.update(size)

def tokenize_file(txt_path: Path, bin_path: Path):
    if not txt_path.exists():
        return
    if bin_path.exists():
        print(f"File {bin_path} already exists, skipping tokenization.")
        return
    print(f"Tokenizing {txt_path} to {bin_path}...")
    enc = tiktoken.get_encoding("gpt2")
    # TinyStories separates stories with <|endoftext|>
    eot = enc.eot_token

    # Read all text
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into stories
    stories = text.split('<|endoftext|>')
    
    # Tokenize each story and append EOT
    all_tokens = []
    for story in tqdm(stories, desc=f"Tokenizing {txt_path.name}"):
        story = story.strip()
        if not story:
            continue
        tokens = enc.encode_ordinary(story)
        tokens.append(eot)
        all_tokens.extend(tokens)

    # Save to binary file
    arr = np.array(all_tokens, dtype=np.uint16) # GPT-2 vocab is ~50257, fits in uint16
    arr.tofile(bin_path)
    print(f"Saved {len(all_tokens):,} tokens to {bin_path}")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for url in URLS:
        filename = url.split('/')[-1]
        txt_path = DATA_DIR / filename
        bin_path = DATA_DIR / filename.replace(".txt", ".bin")
        
        # Download
        download_file(url, txt_path)
        
        # Tokenize
        tokenize_file(txt_path, bin_path)

if __name__ == "__main__":
    main()
