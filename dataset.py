"""A dataset to hold Byte Pair encoded text."""
import torch
from torch.utils.data import Dataset
from encoder import BytePairEncoder

class BPEDataset(Dataset):
  def __init__(self, encoder: BytePairEncoder, text: str, context: int):
    self.corpus = encoder.encode(text)
    self.encoder = encoder
    self.context = context
    self.corpus = torch.tensor(self.corpus, dtype=torch.long)
  def __len__(self):
    return len(self.corpus)-self.context
  def __getitem__(self, idx):
    return self.corpus[idx:idx+self.context], self.corpus[idx+1:idx+self.context+1] # expected output is inout window shifted one right