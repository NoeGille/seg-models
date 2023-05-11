from dataset_florian import FashionMNISTDataset
from torch.utils.data import DataLoader, Subset
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
BATCH_SIZE = 16
INPUT_SIZE = (224, 224, 1)

# LOADING DATASETS

train_data_easy = torch.load('datasets/train_data_easy1.pt')
train_data_easy2 = torch.load('datasets/train_data_easy2.pt')

# SPLITTING DATASETS FOR CROSS VALIDATION
loader = DataLoader(train_data_easy, batch_size = BATCH_SIZE, shuffle = False)
subset1 = Subset(train_data_easy, range(0, 2000))
subset2 = Subset(train_data_easy, range(0, 2000))
subset1 = DataLoader(train_data_easy, batch_size = BATCH_SIZE, shuffle = False)
subset2 = DataLoader(train_data_easy2, batch_size = BATCH_SIZE, shuffle = False)
print((subset1.dataset[0][0] == subset2.dataset[0][0]).sum().item())
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(train_data_easy2[0][0].squeeze())
fig.add_subplot(1,2,2)
plt.imshow(train_data_easy[0][0].squeeze())
plt.show()
print(len(loader.dataset[0][0].flatten()))
dataloader1 = DataLoader(subset, batch_size = BATCH_SIZE, shuffle = False)
