import matplotlib.pyplot as plt
import numpy as np
from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
import torch
# CONSTANTS
DATASET_PATH = 'datasets/'
dataset_name = "data_len_1000"
# LOAD DATASET
dataset = torch.load(DATASET_PATH + dataset_name + '.pt')

# PLOT 9 samples
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    img, mask = dataset[i]
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(img.squeeze(), cmap='gray')
    ax.imshow(mask.squeeze(), alpha=1)
    ax.axis('off')
plt.show()
