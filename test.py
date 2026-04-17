import torch, pickle
from time import sleep
with open("model.pkl", "rb") as f:
  model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
  encoder = pickle.load(f)
model.to("cpu")
text = "A"
tokens = torch.tensor(encoder.encode(text))
with torch.no_grad():
  while True:
    x = torch.unsqueeze(tokens, dim=0)
    logits = model(x)[0]
    probs = torch.softmax(logits/100, dim=-1)
    choice = torch.unsqueeze(torch.multinomial(probs, num_samples=1)[0, -1], dim=0)
    new_token = choice.tolist()
    tokens = torch.cat((tokens, torch.tensor(new_token)), dim=-1)
    print(encoder.decode(tokens.tolist()))
    sleep(1)