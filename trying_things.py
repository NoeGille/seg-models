from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
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

model = UNETR(img_size = 224, depth=2, skip_connections=[0,1])

# LOAD A UNETR MODEL
model_names = ['unetr_depth1']

for model_name in model_names:
    # LOADING MODEL AND DATA FROM FILES
    checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
    num_classes = checkpoint['num_classes']
    kwargs = checkpoint['kwargs']
    model = checkpoint['model_class'](**kwargs)
    dataset_name = checkpoint['datasets']
    dataset = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=DEVICE)
    
    model.eval()

    sample_image, sample_mask = dataset[0]



# GETTING THE ATTENTION MAPS FROM FIRST BLOCK
attention_maps = model.attention_map(sample_image.unsqueeze(0).to(device=DEVICE)) # dim : (n_samples, nb_of_blocks, nb_heads, 197, 197)
print(attention_maps)
print(attention_maps[0].shape)

# We get rid of the cls_token
first_block_attention_maps = attention_maps[0][0][0][1:, 1:] # dim : (196, 196)
print(first_block_attention_maps.shape)

# PLOTTING THE FIRST ATTENTION MAP
fig = plt.figure(figsize=(12.5, 25))
for i in range(196):
    fig.add_subplot(14, 14, i+1)
    plt.imshow(first_block_attention_maps[i, :].reshape(14, 14).cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
plt.figure(figsize=(12.5, 25))
plt.imshow(sample_image[0, :, :].cpu().detach().numpy(), cmap='gray')
plt.axis('off')
plt.show()

