import torch, pickle
from time import sleep
from encoder import BytePairEncoder
from llm import LLM
import json

with open("encoder.pkl", "rb") as f:
  encoder = pickle.load(f)

with open("model.pkl", "rb") as f:
  old_state = torch.load(f, weights_only=False).state_dict()

with open("config.json", "rb") as f:
  params = json.load(f)

model = LLM(num_tokens=128+len(encoder.token_pairs),
            d_model=params["d_model"],
            max_tokens=params["context_length"],
            attention_layers=params["attn_layers"],
            attention_heads=params["attn_heads"],
            ff_hiddens=params["feedforward_layers"],
            ff_hidden_size=params["feedforward_layer_size"])

model.load_state_dict(old_state)

model.to("cpu")
text = "I"
tokens = torch.tensor(encoder.encode(text))
with torch.no_grad():
  while len(tokens) < 100:
    x = torch.unsqueeze(tokens[-32:], dim=0)
    logits = model(x, mask=True)[0, -1]
    probs = torch.softmax(logits, dim=-1)
    choice = torch.unsqueeze(torch.multinomial(probs, num_samples=1)[-1], dim=0)
    new_token = choice.tolist()
    tokens = torch.cat((tokens, torch.tensor(new_token)), dim=-1)
  print(encoder.decode(tokens.tolist()))
with open("out.txt", "w") as f:
  f.write(encoder.decode(tokens.tolist()))