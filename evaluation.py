from dataset_florian import FashionMNISTDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOADING DATASETS

valid_data_easy = torch.load('datasets/valid_data_easy.pt')
test_dataloader_easy = DataLoader(valid_data_easy, batch_size = 16, shuffle = False)


# LOADING MODEL

checkpoint = torch.load('models/unet1.pt')
input_size = checkpoint['input_size']
num_classes = checkpoint['num_classes']
kwargs = checkpoint['kwargs']

model = UNet(input_size, num_classes, **kwargs)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device=DEVICE)
model.eval()

sample_image, sample_mask = valid_data_easy[0]
sample_image_to_show = sample_image.permute(1, 2, 0)[:,:,0]
plt.figure()
plt.imshow(sample_image_to_show, cmap = 'gray')
plt.figure()
plt.imshow(sample_mask, vmin=0, vmax=10, cmap='tab10')
fig = plt.figure()
with torch.no_grad():
    prediction = model(torch.from_numpy(np.array([sample_image.numpy()])).to(device=DEVICE))
    plt.imshow(prediction.argmax(1).cpu().numpy()[0], vmin=0, vmax=10, cmap='tab10')
    plt.colorbar()
plt.show()




