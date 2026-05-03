import torch
import torch.nn as nn
import torch.nn.functional as functional
import pickle
from torch.utils.data import DataLoader
import json
import os
import time
from encoder import BytePairEncoder
from dataset import BPEDataset
from llm import LLM
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/llm")

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

train_dataset = BPEDataset(encoder,
                          shakespeare,
                          params["context_length"])
batch_per_epoch = len(train_dataset)//params["batch_size"]
train_loader = DataLoader(train_dataset,
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
optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
for epoch in range(params["epochs"]):
  for i, data in enumerate(train_loader):
    start = time.perf_counter_ns()
    x, y = data
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(x)
    loss = loss_func(outputs.transpose(-2, -1), y)
    loss.backward()
    optimizer.step()
    writer.add_scalar("loss", loss.item(), epoch*batch_per_epoch+i)
    print(f"Loss at epoch {epoch+1}/{params["epochs"]}, batch {i}/{batch_per_epoch}: {loss.item()}, time: {round((time.perf_counter_ns()-start)*10**-9, 4)} secs")
  with open(params["model_path"], "wb") as f:
    torch.save(model, f)
writer.close()