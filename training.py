from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from carbontracker.tracker import CarbonTracker
from metrics import LossManager
import segmentation_models_pytorch as smp
from CamusEDImageDataset1 import CamusEDImageDataset
from metrics import LossManager



# CONSTANTS

DATASET_PATH = 'datasets/'
MODEL_PATH = 'models/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
BATCH_SIZE = 16
INPUT_SIZE = (224, 224, 3)

def continue_training(model_name, epochs, learning_rate, dataset_name, freezing_function:callable = lambda x: None, new_model_name = None, validation_loss=True):
    '''load and train a pretrained model on a specified dataset
    freezing_function : function that takes the model as input and freeze some parameters (The frozen parameters are not updated during training)
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
    valid_data = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
    valid_dataloader= DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False)
    loss_manager = LossManager()

    # TRAINING
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for (train_img, train_mask), (valid_img, valid_mask) in tqdm(zip(train_dataloader, valid_dataloader)):
            # Calculate loss on validation set
            if validation_loss:
                with torch.no_grad():
                    valid_img = valid_img.to(device=DEVICE)
                    valid_mask = valid_mask.to(device=DEVICE)
                    
                    loss = criterion(model(valid_img), valid_mask)
                    loss_manager.add(loss.item())

            train_img = train_img.to(device=DEVICE)
            train_mask = train_mask.to(device=DEVICE) # dim : (batch_size, 224, 224)
            # prediction
            mask_pred = model(train_img)  # dim : (batch_size, 10, 224, 224)

            # Calculate loss
            loss = criterion(mask_pred, train_mask)
            
            
            # backward
            optimizer.zero_grad()
            loss.backward()

            loss_manager.add(loss.item())

            # gradient descent or adam step
            optimizer.step()
        loss_manager.epoch_end()
    
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
    file = open('results/loss/' + f'{new_model_name}_loss.txt', 'w')
    for elt in loss_manager.losses:
        file.write(str(float(elt)) + '\n')
    file.close()



def train(model_class, kwargs, learning_rate, epochs, model_name, dataset_name, validation_loss=True):
    '''Train a model on a specified dataset
    Return the trained model and the loss of the last epoch'''

    # LOADING DATASET
    print(f'Loading dataset {dataset_name}')
    train_data = torch.load(DATASET_PATH + 'train_' + dataset_name + '.pt')
    train_dataloader= DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)
    valid_data = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
    valid_dataloader= DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False)
    # CREATE MODEL AND OPTIMIZER
    print(f'Creating model {model_class.__class__.__name__} with {kwargs}')
    model = model_class(**kwargs).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    print(f'Training {model.__class__.__name__} with {kwargs} for {epochs} epochs on {dataset_name}')
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    #tracker = CarbonTracker(epochs=epochs, log_dir="logs/" + model_name)
    loss_manager = LossManager()

    # TRAINING
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        #tracker.epoch_start()
        for (train_img, train_mask), (valid_img, valid_mask) in tqdm(zip(train_dataloader, valid_dataloader)):
            # Calculate loss on validation set
            if validation_loss:
                with torch.no_grad():
                    valid_img = valid_img.to(device=DEVICE)
                    valid_mask = valid_mask.to(device=DEVICE)
                    loss = criterion(model(valid_img), valid_mask.long())
                    loss_manager.add(loss.item())

            train_img = train_img.to(device=DEVICE)
            train_mask = train_mask.to(device=DEVICE) # dim : (batch_size, 224, 224)
            # prediction
            mask_pred = model(train_img)  # dim : (batch_size, 10, 224, 224)
            # Calculate loss
            loss = criterion(mask_pred, train_mask.long())
            
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            loss_manager.add(loss.item())

            # gradient descent or adam step
            optimizer.step()

        #tracker.epoch_end()
        loss_manager.epoch_end()

    #tracker.stop()
        
    # SAVING MODEL 
    # <!> Every arguments of the model initialization will be saved in kwargs dictionary<!>
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
    file = open('results/loss/' + f'{model_name}_loss.txt', 'w')
    for elt in loss_manager.losses:
        file.write(str(float(elt)) + '\n')
    file.close()

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

def freeze_smp(model : smp.Unet, n):
    '''Freeze the first n layers of the model. Only works with Unet models from the segmentation models pytorch library.'''
    encoder_features = []
    for name, module in model.named_modules():
        splits = name.split('.')
        if name.count('.') < 2:
            continue
        if splits[0] == 'encoder' and (isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d)):
            encoder_features.append((name, module))
        if splits[0] == 'decoder':
            if splits[1] == 'blocks':
                if len(model.decoder.blocks) - int(splits[2]) <= n:
                    for param in module.parameters():
                        param.requires_grad = False
    # Freeze the first layers of the encoder until it reaches the n-th maxpooling layer
    counter = 0
    for name, module in encoder_features:
        if counter < n:
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, nn.MaxPool2d):
                counter += 1
    print("Number of parameters updated: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

# LIST OF PARAMETERS

'''list of every model, model parameters and training parameters to train

training_params = [[model class, kwargs, learning rate, epochs, model name, dataset name], [...], [...
Example:
training_params = [
    [UNETR, {'depth':12, 'skip_connections':[2,5,8,11], 'pretrained_name': 'vit_base_patch16_224', 'num_classes':NUM_CLASSES}, 0.0001, 20, 'unetr_depth12_camus', 'data_camus1'],
    [UNet, {'depth':5, 'num_classes':NUM_CLASSES, 'input_size': INPUT_SIZE}, 0.0001, 20, 'unet_depth5_camus', 'data_camus1'],
    [smp.Unet, {'encoder_name':'vgg16',
                    'encoder_weights':'imagenet',
                    'in_channels':3,
                    'encoder_depth':5,
                    'decoder_channels':(256, 128, 64, 32, 16),
                    'classes':10}, 0.0001, 20, 'unet_depth5', 'dataset_rgb'],
]'''

training_params = [
    [UNETR, {'depth':2, 'skip_connections':[0,1], 'pretrained_name': 'vit_base_patch16_224', 'num_classes':NUM_CLASSES}, 0.0001, 20, 'unetr_depth2_sc01_pre_lr', 'data_rgb_b_noise_len_10k'],
   ]

# For pre-trained models
# continue_training_params = [[model name, epochs, learning rate, dataset name, freezing function, new model name], [...], [...]
# Example:
# continue_training_params = [
#     ['unet_pre4', 25, 0.0001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=True), 'unet_pre4_e25_size1000'],
# Freezing functions can be defined in this file and allows you to freeze some parameters in the model during training
# For example, the function UNet_freeze allows you to freeze the encoder, decoder and skip connections of the UNet model   

continue_training_params = [
    ['unet_vgg_depth5_pre_e5', 50, 0.0001, 'data_hard_b_len1000', lambda model: freeze_smp(model, 5), 'unet_vgg_depth5_pre_e55_len1000_b'],
]


for i, params in enumerate(training_params):
    print(f'Training model {i+1}/{len(training_params)}')
    train(*params)

for i, params in enumerate(continue_training_params):
    print(f'Training model {i+1}/{len(continue_training_params)}')
    continue_training(*params)

'''
Training parameters for comuting the receptive field TODO

    [smp.Unet, {'encoder_name':'vgg16',
                    'encoder_weights':'imagenet',
                    'in_channels':3,
                    'encoder_depth':4,
                    'decoder_channels':(256, 128, 64, 32,),
                    'classes':10}, 0.0001, 20, 'unet_depth4_10k', 'data_rgb_b_noise_len_10k'],
    [smp.Unet, {'encoder_name':'vgg16',
                    'encoder_weights':'imagenet',
                    'in_channels':3,
                    'encoder_depth':3,
                    'decoder_channels':(256, 128, 64,),
                    'classes':10}, 0.0001, 20, 'unet_depth3_10k', 'data_rgb_b_noise_len_10k'],
    [smp.Unet, {'encoder_name':'vgg16',
                    'encoder_weights':'imagenet',
                    'in_channels':3,
                    'encoder_depth':2,
                    'decoder_channels':(256, 128),
                    'classes':10}, 0.0001, 20, 'unet_depth2_10k', 'data_rgb_b_noise_len_10k'],
'''

'''['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=True), 'unet_frozen_nobottle_1'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=True), 'unet_frozen_nobottle_2'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=True), 'unet_frozen_nobottle_3'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=True), 'unet_frozen_nobottle_4'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=False), 'unet_frozen_1'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=False), 'unet_frozen_2'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=False), 'unet_frozen_3'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=False), 'unet_frozen_4'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [], freeze_bottleneck=False), 'unet_frozen_all'],
    ['unet_pre4', 25, 0.001, 'data_hard_b_len1000', lambda model: UNet_freeze(model, [1,2,3,4], freeze_bottleneck=False), 'unet_frozen_bottle'],'''
