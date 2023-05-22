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
    layers = [layer - 1 for layer in layers]
    for name, module in model.named_modules():
        splits = name.split('.')
        print(name, end=' ')
        if (len(splits) > 2 and int(splits[1]) in layers) or (splits[0] == 'bottleneck' and freeze_bottleneck):
            for param in module.parameters():
                param.requires_grad = False
                print('FROZEN', end=' ')
        print()


# GENERATE DATASETS


