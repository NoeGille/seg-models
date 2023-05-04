from dataset_florian import FashionMNISTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch


# CONSTANTS

DATASET_PATH = 'datasets/'
MODEL_PATH = 'models/'
MODEL_NAME = 'unet1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001
INPUT_SIZE = (224, 224, 1)


# LOADING DATASETS

train_data_easy = torch.load(DATASET_PATH + 'train_data_easy.pt')
train_dataloader_easy = DataLoader(train_data_easy, batch_size = BATCH_SIZE, shuffle = False)

# CREATING MODEL

# Use **kwargs to pass arguments to the model for saving
kwargs = {'depth':3, 'dilation':2}
model = UNet(INPUT_SIZE, NUM_CLASSES, **kwargs).to(DEVICE)

print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# TRAINING

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    for img, mask in tqdm(train_dataloader_easy):
        
        img = img.to(device=DEVICE)
        mask = mask.to(device=DEVICE) # dim : (batch_size, 224, 224)
        # prediction
        mask_pred = model(img)  # dim : (batch_size, 10, 224, 224)
        # Calculate loss
        loss = criterion(mask_pred, mask)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

# SAVING MODEL 
# <!> Every arguments of the model initialization must be saved in kwargs dictionary<!>
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(
    {
    'input_size':INPUT_SIZE,
    'epochs':EPOCHS,
    'learning_rate':LEARNING_RATE,
    'batch_size':BATCH_SIZE,
    'num_classes':NUM_CLASSES,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':loss,
    'kwargs':kwargs
    }, MODEL_PATH + MODEL_NAME + '.pt'
)
