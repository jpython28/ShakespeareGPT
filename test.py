import torch, pickle
from time import sleep
from llm import LLM
import json

with open("encoder.pkl", "rb") as f:
  encoder = pickle.load(f)

with open("config.json", "r") as f:
  params = json.load(f)
model = LLM(num_tokens=128+len(encoder.token_pairs),
            d_model=params["d_model"],
            max_tokens=params["context_length"],
            attention_layers=params["attn_layers"],
            attention_heads=params["attn_heads"],
            ff_hiddens=params["feedforward_layers"],
            ff_hidden_size=params["feedforward_layer_size"])

with open("model.pkl", "rb") as f:
  old_state = pickle.load(f).state_dict()
  if "positional_encodings" in old_state:
    old_state["positional_encodings"] = old_state["positional_encodings"].squeeze(0)
  model.load_state_dict(old_state, strict=False)

model.to("cpu")
text = "thou"
tokens = torch.tensor(encoder.encode(text))
with torch.no_grad():
  while True:
    x = torch.unsqueeze(tokens, dim=0)
    logits = model(x, mask=False)[0]
    probs = torch.softmax(logits, dim=-1)
    choice = torch.unsqueeze(torch.multinomial(probs, num_samples=1)[0, -1], dim=0)
    new_token = choice.tolist()
    tokens = torch.cat((tokens, torch.tensor(new_token)), dim=-1)
    print(encoder.decode(tokens.tolist()))
    sleep(1)