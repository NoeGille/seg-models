from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import wandb
from carbontracker.tracker import CarbonTracker



# CONSTANTS

DATASET_PATH = 'datasets/'
MODEL_PATH = 'models/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NUM_CLASSES = 10
BATCH_SIZE = 16
INPUT_SIZE = (224, 224, 1)

def continue_training(model_name, epochs, learning_rate, dataset_name, freezing_function:callable = lambda x: None, new_model_name = None):
    '''load and train a pretrained model on a specified dataset
    A freezing functions can be specified to freeze some parameters of the models
    new_model_name : if specified, the model will be saved with this name'''
    print(f'Loading model {model_name}')

    # LOADING MODEL
    checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
    kwargs = checkpoint['kwargs']
    model_class = checkpoint['model_class']
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # FREEZING SOME PARAMETERS
    freezing_function(model)

    # LOADING DATASET
    print(f'Loading dataset {dataset_name}')
    train_data = torch.load(DATASET_PATH + 'train_' + dataset_name + '.pt')
    train_dataloader= DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)

    tracker = CarbonTracker(epochs=epochs, log_dir="logs/" + model_name)
    
    # TRAINING
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        tracker.epoch_start()
        for img, mask in tqdm(train_dataloader):
            
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
        tracker.epoch_end()
        
    tracker.stop()
    
    # SAVING MODEL
    if new_model_name is None:
        new_model_name = model_name
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
        }, MODEL_PATH + new_model_name + '.pt'
    )




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
    
    #tracker = CarbonTracker(epochs=epochs, log_dir="logs/" + model_name)
    
    # TRAINING
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        #tracker.epoch_start()
        for img, mask in tqdm(train_dataloader):
            
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
        #tracker.epoch_end()

        
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

def UNet_freeze(model, layers:list, freeze_bottleneck:bool = False):
    '''Freeze all specified layers of the UNet model
    (In this case layers are the depth of the model)
    Both encoder, skip connections and decoder part of the layer are frozen
    The bottleneck can also be frozen with boolean freeze_bottleneck parameter
    '''
    layers = [layer - 1 for layer in layers]
    for name, module in model.named_modules():
        splits = name.split('.')
        if (len(splits) > 2 and int(splits[1]) in layers) or (splits[0] == 'bottleneck' and freeze_bottleneck):
            for param in module.parameters():
                param.requires_grad = False

# LIST OF PARAMETERS

'''list of every model, model parameters and training parameters to train
This allows us to train multiple models in the same script
training_params = [[model class, kwargs, learning rate, epochs, model name, dataset name], [...], [...'''


# <!> Look at pretrained model for UNETR to save some time <!>
training_params = [
    [UNETR, {'depth':1, 'skip_connections':[0], 'pretrained_name': 'vit_base_patch16_224', 'num_classes':NUM_CLASSES}, 0.001, 25, 'unetr_depth1', 'data_rgb_b_len_1000'],
    #[UNETR, {'depth':2, 'skip_connections':[1], 'pretrained_name': 'vit_base_patch16_224'}, 0.001, 25, 'unetr_depth2', 'data_hard_b_len1000'],
    #[UNETR, {'depth':4, 'skip_connections':[3], 'pretrained_name': 'vit_base_patch16_224'}, 0.001, 100, 'unetr_depth4', 'data_hard_b_len1000'],
]

'''
Training parameters for comuting the receptive field TODO
'''


# For pre-trained models
continue_training_params = [
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=True), 'unet_frozen_nobottle_1'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=True), 'unet_frozen_nobottle_2'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=True), 'unet_frozen_nobottle_3'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=True), 'unet_frozen_nobottle_4'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=False), 'unet_frozen_1'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=False), 'unet_frozen_2'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=False), 'unet_frozen_3'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=False), 'unet_frozen_4'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [], freeze_bottleneck=False), 'unet_frozen_all'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3,4], freeze_bottleneck=False), 'unet_frozen_bottle'],
]

for i, params in enumerate(training_params):
    print(f'Training model {i+1}/{len(training_params)}')
    train(*params)

'''for i, params in enumerate(continue_training_params):
    print(f'Training model {i+1}/{len(continue_training_params)}')
    continue_training(*params)'''