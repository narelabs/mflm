import os
import sys
import numpy as np
import tiktoken

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'wikitext2')
    os.makedirs(data_dir, exist_ok=True)
    
    enc = tiktoken.get_encoding("gpt2")
    
    try:
        from datasets import load_dataset
        print("Using datasets library...", flush=True)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        for split in ['train', 'validation', 'test']:
            print(f"Processing {split} split...", flush=True)
            tokens = []
            for item in dataset[split]:
                text = item['text']
                if text.strip():
                    tokens.extend(enc.encode_ordinary(text))
            
            tokens = np.array(tokens, dtype=np.uint16)
            out_name = 'valid' if split == 'validation' else split
            out_path = os.path.join(data_dir, f'{out_name}.bin')
            tokens.tofile(out_path)
            print(f"Saved {len(tokens)} tokens to {out_path}", flush=True)

    except ImportError:
        print("datasets library not found, falling back to urllib...", flush=True)
        import urllib.request
        import zipfile
        import io
        
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        print(f"Downloading {url}...", flush=True)
        response = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            for filename in z.namelist():
                if filename.endswith('.raw'):
                    split_name = filename.split('.')[-2].split('-')[-1]
                    if split_name not in ['train', 'valid', 'test']:
                        continue
                    
                    print(f"Processing {split_name} split...", flush=True)
                    text = z.read(filename).decode('utf-8')
                    tokens = enc.encode_ordinary(text)
                    tokens = np.array(tokens, dtype=np.uint16)
                    out_path = os.path.join(data_dir, f'{split_name}.bin')
                    tokens.tofile(out_path)
                    print(f"Saved {len(tokens)} tokens to {out_path}", flush=True)

if __name__ == "__main__":
    main()
