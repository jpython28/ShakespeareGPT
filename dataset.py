import torch
from torch.utils.data import Dataset
from tokenizers import BytePairEncoder

def one_hot_encode(tokens, num_tokens: int):
  return torch.eye(num_tokens)[tokens]

class BPEDataset(Dataset):
  def __init__(self, encoder: BytePairEncoder, text: str, context: int):
    self.corpus = encoder.encode(text)
    self.encoder = encoder
    self.context = context
    self.corpus = torch.tensor(self.corpus, dtype=torch.long)
  def __len__(self):
    return len(self.corpus)-self.context
  def __getitem__(self, idx):
    return self.corpus[idx:idx+self.context], one_hot_encode(self.corpus[idx+1:idx+self.context+1], 128+len(self.encoder.token_pairs))