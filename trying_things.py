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


def UNet_freeze(model, layers:list, freeze_bottleneck:bool = False):
    '''Freeze all specified layers of the UNet model
    (Here layers are the depth of the model)'''
    for name, module in model.named_modules():
        splits = name.split('.')
        if (len(splits) > 2 and int(splits[1]) in layers) or (splits[0] == 'bottleneck' and freeze_bottleneck):
            for param in module.parameters():
                param.requires_grad = False


model = UNet(input_size=INPUT_SIZE, num_classes=NUM_CLASSES, depth=4)
UNet_freeze(model, [1, 2, 3])
print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
