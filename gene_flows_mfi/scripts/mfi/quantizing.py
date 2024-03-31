#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[ ]:


import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from src.pytorch_generative.pytorch_generative import models
from src.dataset import load_dataset
from src.experiment import eval_run


# In[ ]:


seed = 4
torch.manual_seed(seed)
np.random.seed(seed)


# In[ ]:


train_path = r"../../data/processed/train.csv"
test_path = r"../../data/processed/test.csv"

train_data, val_data, test_data = load_dataset(
    train_path=train_path,
    test_path=test_path,
)

batch_size = 100000
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

pre_data = pd.read_csv(train_path).head(100)
df = pre_data.drop("Index", axis=1)
subset_data = df.to_numpy()
subset_data = np.where(subset_data > 0, 1, 0)
subset_data = subset_data[:, np.newaxis, np.newaxis, :]
print(subset_data.shape)
subset_train_loader = DataLoader(subset_data, batch_size, shuffle=True)


# In[ ]:


sequence_length = 1000
# python3 made_genes_data.py --device cuda --log_dir ../tmp/run/made_sweep_learnings --batch_size 512 --use_wandb True --n_epochs 100 --weight_decay 0.00001 --hidden_dims 2500 --num_layers 2 --n_masks 1 --lr 0.001 --step_size 10
log_dir = "../../output/networks/"
val_epoch = 84


# In[ ]:


# load the model
hidden_dims = 2500
num_layers = 2
n_masks = 1
device = "cuda"
model = models.MADE(
    input_dim=sequence_length,
    hidden_dims=[hidden_dims] * num_layers,
    n_masks=n_masks,
    device=device,
)
model = model.to(device)


# In[ ]:


# get_ipython().system('pwd')
# ! ls ../../tmp/run/made_sweep_learnings/


# In[ ]:


checkpoint = torch.load(os.path.join(log_dir, f"trainer_state_{val_epoch}.ckpt"))
model_state_dict = {
    k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
}
model.load_state_dict(model_state_dict)
model = model.eval()
model = model.to(device)


# In[ ]:


for param in model.parameters():
    print(param.shape)
    print(param.numel())


# In[ ]:


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {pytorch_total_params}")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {pytorch_total_params}")


# In[ ]:


# these usually get initalized on the first run
model._c = 1
model._h = 1
model._w = sequence_length


# In[ ]:


import time


# Function to calculate evaluation time
def evaluate(loader, model, device):
    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    result = eval_run(loader, model, device)
    end_time = time.time()
    evaluation_time = end_time - start_time
    return result, evaluation_time


# In[ ]:


# Evaluate train_loader
train_result, train_time = evaluate(train_loader, model, device)
# Evaluate val_loader
val_result, val_time = evaluate(val_loader, model, device)

# Evaluate test_loader
test_result, test_time = evaluate(test_loader, model, device)

print("Train Evaluation Time:", round(train_time, 5), "seconds")
print("Validation Evaluation Time:", round(val_time, 5), "seconds")
print("Test Evaluation Time:", round(test_time, 5), "seconds")

print("Train Result:", round(train_result.item(), 5))
print("Validation Result:", round(val_result.item(), 5))
print("Test Result:", round(test_result.item(), 5))


# In[ ]:


import torch
import torchvision

# Set model to eval mode
model.eval()

# Specify quantization configuration
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Prepare the model for static quantization
model = torch.quantization.prepare(model, inplace=False)

# Calibrate the model
# Note: replace 'data' with your representative dataset
for data in train_loader:
    model(data.to(device))

# Convert to quantized model
model_quant = torch.quantization.convert(model, inplace=False)

torch.save(
    model_quant,
    os.path.join(
        "..",
        "..",
        "outputs",
        "networks",
        "quantized",
        "model_quant.ckpt",
    ),
)


# In[ ]:


# Evaluate train_loader
train_result, train_time = evaluate(train_loader, model, device)
# Evaluate val_loader
val_result, val_time = evaluate(val_loader, model, device)

# Evaluate test_loader
test_result, test_time = evaluate(test_loader, model, device)

print("quantized: ")

print("Train Evaluation Time:", round(train_time, 5), "seconds")
print("Validation Evaluation Time:", round(val_time, 5), "seconds")
print("Test Evaluation Time:", round(test_time, 5), "seconds")

print("Train Result:", round(train_result.item(), 5))
print("Validation Result:", round(val_result.item(), 5))
print("Test Result:", round(test_result.item(), 5))


# In[ ]:


# create a quantized model instance
model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.quint8,
)  # the target dtype for quantized weightsx

torch.save(
    model_int8,
    os.path.join(
        "..",
        "..",
        "outputs",
        "networks",
        "quantized",
        "model_int8.ckpt",
    ),
)


# In[ ]:

print("quint8: ")
# Evaluate train_loader
train_result, train_time = evaluate(train_loader, model, device)
print("Train Evaluation Time:", round(train_time, 5), "seconds")
print("Train Result:", round(train_result.item(), 5))

# Evaluate val_loader
val_result, val_time = evaluate(val_loader, model, device)
print("Validation Evaluation Time:", round(val_time, 5), "seconds")
print("Validation Result:", round(val_result.item(), 5))

# Evaluate test_loader
test_result, test_time = evaluate(test_loader, model, device)
print("Test Evaluation Time:", round(test_time, 5), "seconds")
print("Test Result:", round(test_result.item(), 5))


# In[37]:
