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



def train(model_class, kwargs, learning_rate, epochs, model_name, dataset_name):
    '''Train a model on a specified dataset
    Return the trained model and the loss of the last epoch'''

    # LOADING DATASET
    print(f'Loading dataset {dataset_name}')
    train_data = torch.load(DATASET_PATH + 'train_' + dataset_name + '.pt')
    train_dataloader= DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)

    # CREATE MODEL AND OPTIMIZER
    print(f'Creating model {model_class.__class__.__name__} with {kwargs}')
    model = model_class(**kwargs).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    print(f'Training {model.__class__.__name__} with {kwargs} for {epochs} epochs on {dataset_name}')
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # TRAINING
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        i = 0
        for img, mask in tqdm(train_dataloader):
            i+=1
            
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
    # <!> Every arguments of the model initialization will be saved in kwargs dictionary<!>
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(
    {   
        'input_size':INPUT_SIZE,
        'epochs':epochs,
        'learning_rate':learning_rate,
        'batch_size':BATCH_SIZE,
        'num_classes':NUM_CLASSES,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
        'datasets': dataset_name,
        'model_class': model_class,
        'kwargs':kwargs
        }, MODEL_PATH + model_name + '.pt'
    )


# LIST OF PARAMETERS

'''list of every model, model parameters and training parameters to train
This allows us to train multiple models in the same script
training_params = [[model class, kwargs, learning rate, epochs, model name, dataset name], [...], [...'''
training_params = [
    [UNETR, {'depth':12, 'p':0.25, 'attn_p':0.25}, 0.001, 25, 'unetr_d8', 'data_hard_len100'],
]

UNETR()
# TRAINING

for i, params in enumerate(training_params):
    print(f'Training model {i+1}/{len(training_params)}')
    train(*params)

