import torch
import torch.nn as nn

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
    positional_encodings = torch.zeros((self.max_tokens, self.d_model))
    assert self.max_tokens%2==0
    for pos in range(max_tokens):
      positional_encodings[pos, 0::2] = torch.sin(torch.arange(d_model // 2) / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
      positional_encodings[pos, 1::2] = torch.cos(torch.arange(d_model // 2) / (10000 ** (torch.arange(0, d_model, 2) / d_model)))
    self.register_buffer("positional_encodings", positional_encodings)
    self.multiheadattentions = nn.ModuleList(list([nn.MultiheadAttention(self.d_model, self.attention_heads, batch_first=True) for _ in range(self.attention_layers)]))
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
    self.layer_norms1 = nn.ModuleList()
    self.layer_norms2 = nn.ModuleList()

    for i in range(self.attention_layers):
      self.layer_norms1.append(nn.LayerNorm(self.d_model))
      self.layer_norms2.append(nn.LayerNorm(self.d_model))

  def forward(self, x):
    tokens = x
    embedded = self.embedding(tokens)
    embedded = embedded + self.positional_encodings[:embedded.shape[1]].repeat(embedded.shape[0], 1, 1) # type: ignore
    for i in range(self.attention_layers):
      q, k, v = embedded, embedded, embedded
      embedded = embedded + self.multiheadattentions[i](q, k, v, need_weights=False)[0]
      embedded = self.layer_norms1[i](embedded)
      embedded = embedded + self.feed_forwards[i](embedded)
      embedded = self.layer_norms2[i](embedded)
    logits = torch.matmul(embedded, self.embedding.weight.T)
    return logits