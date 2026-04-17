import torch
import torch.nn as nn
import torch.nn.functional as functional
import pickle
from torch.utils.data import DataLoader
from llm import LLM
from tokenizers import BytePairEncoder
from dataset import BPEDataset
import json
import os
import time

with open("config.json", "r") as f:
  params = json.load(f)

with open ("shakespeare.txt", "r") as f:
  shakespeare = f.read()

if not os.path.exists(params["encoder_path"]):
  encoder = BytePairEncoder(params["vocab_size"], shakespeare)
  with open(params["encoder_path"], "wb") as f:
    pickle.dump(encoder, f)
else:
  with open(params["encoder_path"], "rb") as f:
    encoder = pickle.load(f)

train_loader = DataLoader(BPEDataset(encoder, 
                                     shakespeare, 
                                     params["context_length"]), 
                          batch_size=params["batch_size"], 
                          shuffle=True)

model = LLM(num_tokens=128+len(encoder.token_pairs), 
            d_model=params["d_model"], 
            max_tokens=params["context_length"], 
            attention_layers=params["attn_layers"], 
            attention_heads=params["attn_heads"], 
            ff_hiddens=params["feedforward_layers"], 
            ff_hidden_size=params["feedforward_layer_size"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {str(device).upper()}")
model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
for epoch in range(2):
  for i, data in enumerate(train_loader):
    start = time.perf_counter_ns()
    x, y = data
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(x)
    loss = loss_func(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Loss at epoch {epoch}, batch {i}: {loss.item()}, time: {round((time.perf_counter_ns()-start)*10**-9)} secs")
with open(params["model_path"], "wb") as f:
  pickle.dump(model, f)

