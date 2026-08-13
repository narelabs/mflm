import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        # We use uint16 for tiktoken GPT2 encoding since max vocab is ~50257
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        # Total tokens minus seq_len (we need seq_len + 1 for targets)
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Read seq_len + 1 tokens
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def get_dataloader(bin_path: str, seq_len: int, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = TokenizedDataset(bin_path, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
