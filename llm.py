"""The main pytorch language model class"""

import torch
import torch.nn as nn
from math import sin, cos

class LLM(nn.Module):
  def __init__(self, num_tokens: int, d_model: int, max_tokens: int, attention_layers: int, attention_heads: int, ff_hiddens: int, ff_hidden_size: int):
    super().__init__()
    self.max_tokens = max_tokens
    self.num_tokens = num_tokens
    self.d_model = d_model
    self.attention_layers = attention_layers
    self.attention_heads = attention_heads
    self.ff_hiddens = ff_hiddens
    self.ff_hidden_size = ff_hidden_size
    self.embedding = nn.Embedding(self.num_tokens, self.d_model)

    attn_mask = ~torch.tril(torch.full((self.max_tokens, self.max_tokens), True)) # Prevents earlier tokens from attending to earlier ones
    self.register_buffer("attn_mask", attn_mask)

    # initialize sinusoidal positional encoding, gives embeddings a sense of position
    positional_encodings = torch.zeros((self.max_tokens, self.d_model))
    for pos in range(max_tokens):
      positional_encodings[pos, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
      positional_encodings[pos, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
    self.register_buffer("positional_encodings", positional_encodings)

    # Initialize a ModuleList of MultiHeadedAttention blocks
    self.multiheadattentions = nn.ModuleList(list([nn.MultiheadAttention(self.d_model, self.attention_heads, batch_first=True) for _ in range(self.attention_layers)]))
    
    # Initialize a ModuleList of FeedForward Neural networks
    self.feed_forwards = nn.ModuleList()
    for _ in range(self.attention_layers):
      feed_forward = nn.Sequential()
      feed_forward.append(nn.Linear(self.d_model, self.ff_hidden_size))
      feed_forward.append(nn.ReLU())
      for _ in range(self.ff_hiddens):
        feed_forward.append(nn.Linear(self.ff_hidden_size, self.ff_hidden_size))
        feed_forward.append(nn.ReLU())
      feed_forward.append(nn.Linear(self.ff_hidden_size, self.d_model))
      self.feed_forwards.append(feed_forward)
    
    # Initialize LayerNorms
    self.layer_norms1 = nn.ModuleList()
    self.layer_norms2 = nn.ModuleList()
    for i in range(self.attention_layers):
      self.layer_norms1.append(nn.LayerNorm(self.d_model))
      self.layer_norms2.append(nn.LayerNorm(self.d_model))

  def forward(self, x, mask=True):
    """Forward Pass"""
    tokens = x
    embedded = self.embedding(tokens) # Embed tokens

    # Apply positional encoding
    pos = self.positional_encodings[:embedded.shape[1]]
    embedded = embedded + pos.unsqueeze(0)


    for i in range(self.attention_layers):
      residual = embedded
      embedded = self.layer_norms1[i](embedded) # Apply layer normalization
      embedded = residual + self.multiheadattentions[i](embedded, embedded, embedded, need_weights=False, attn_mask = self.attn_mask[:embedded.shape[1], :embedded.shape[1]] if mask else None)[0] # Apply attention
      residual = embedded
      embedded = self.layer_norms2[i](embedded) # Apply layer normalization
      embedded = residual + self.feed_forwards[i](embedded)
    logits = torch.matmul(embedded, self.embedding.weight.T) # Unembed tokens into logits
    return logits