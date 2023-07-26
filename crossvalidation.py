'''This code is very close to train.py'. 
Given a model name and a dataset name, train it and evaluate it with cross validation'''

from evaluation import evaluate
from torch.utils.data import random_split, ConcatDataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import UNet, UNETR
from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
from CamusEDImageDataset1 import CamusEDImageDataset
import segmentation_models_pytorch as smp



DATASET_PATH = 'datasets/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 16
NUM_CLASSES = 10
MODEL_PATH = 'models/'
RESULT_PATH = 'results/camus/'

def save_metrics(metrics_mean, metrics_var, model_name):
    '''Save metrics and various informations about the model'''
    accuracy, precision, recall, dice_score = metrics_mean
    accuracy_var, precision_var, recall_var, dice_score_var = metrics_var
    checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
    kwargs = checkpoint['kwargs']
    datasets = checkpoint['datasets']

    file = open(RESULT_PATH + f'{model_name}.txt', 'w')
    file.write(f'Kwargs : {kwargs}\n')
    file.write(f'Dataset : {datasets}\n')
    file.write(f'Number of epochs : {checkpoint["epochs"]}\n')
    file.write(f'Precision : {precision:.8f}\n')
    file.write(f'Recall : {recall:.8f}\n')
    file.write(f'Dice score : {dice_score:.8f}\n')
    file.write(f'Precision variance : {precision_var:.8f}\n')
    file.write(f'Recall variance : {recall_var:.8f}\n')
    file.write(f'Dice score variance : {dice_score_var:.8f}\n')
    file.close()

def cross_validation(model_class, kwargs, learning_rate, epochs, model_name, dataset_name):
    mean_metrics_list = []
    
    # LOADING DATA
    k = 5
    dataset = torch.load(DATASET_PATH + dataset_name + '.pt')
    splits = random_split(dataset, [len(dataset)//k for i in range(k)])
    
    for i in range(k):
        print(f'Split {i+1}/{k}')
        train_data = ConcatDataset([splits[j] for j in range(k) if j != i])
        valid_data = splits[i]
        
        train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)
        test_dataloader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False)
        
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

            for img, mask in tqdm(train_dataloader):
                
                img = img.to(device=DEVICE)
                mask = mask.to(device=DEVICE) # dim : (batch_size, 224, 224)
                # prediction
                mask_pred = model(img)  # dim : (batch_size, 10, 224, 224)
                # Calculate loss
                loss = criterion(mask_pred, mask.long())

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
                
        mean_metrics, var_metrics = evaluate(model=model, dataset=valid_data)
        mean_metrics_list.append(mean_metrics)
        
    mean = np.array(mean_metrics_list).mean(axis=0)
    var = np.array(mean_metrics_list).var(axis=0)

    # SAVE MODEL 
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
    
    # SAVE METRICS
    save_metrics(mean, var, model_name)

def cross_validation_with_pre_training(model_name, epochs, learning_rate, dataset_name, freezing_function:callable = lambda x: None, new_model_name = None):
    mean_metrics_list = []
    
    # LOADING DATA
    k = 5
    dataset = torch.load(DATASET_PATH + dataset_name + '.pt')
    splits = random_split(dataset, [len(dataset)//k for i in range(k)])
    
    # LOADING MODEL FROM FILES
    
    
    for i in range(k):
        print(f'Split {i+1}/{k}')
        train_data = ConcatDataset([splits[j] for j in range(k) if j != i])
        valid_data = splits[i]
        
        train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = False)
        test_dataloader = DataLoader(valid_data, batch_size = BATCH_SIZE, shuffle = False)
        
        # CREATE MODEL AND OPTIMIZER
        print(f"Loading model {model_name}")
        checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
        num_classes = checkpoint['num_classes']
        kwargs = checkpoint['kwargs']
        model_class = checkpoint['model_class']
        model = model_class(**kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Creating model {model_class.__class__.__name__} with {kwargs} and loading weights')

        print(f'Freezing layers')
        freezing_function(model)
        
        print(f'Training {model.__class__.__name__} with {kwargs} for {epochs} epochs on {dataset_name}')
        print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        # TRAINING
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')

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
        mean_metrics, var_metrics = evaluate(model=model, dataset=valid_data)
        mean_metrics_list.append(mean_metrics)
        
    mean = np.array(mean_metrics_list).mean(axis=0)
    var = np.array(mean_metrics_list).var(axis=0)

    # SAVE MODEL 
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
        }, MODEL_PATH + new_model_name + '.pt'
    )
    
    # SAVE METRICS
    save_metrics(mean, var, new_model_name)
        

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

if __name__ == "__main__":
    parameters = [
        [UNet, {'depth':1, 'input_size':INPUT_SIZE}, 0.0001, 20, 'unet_depth1_camus_novgg', 'data_camus1'],
        [UNet, {'depth':2, 'input_size':INPUT_SIZE}, 0.0001, 20, 'unet_depth2_camus_novgg', 'data_camus1'],
        [UNet, {'depth':3, 'input_size':INPUT_SIZE}, 0.0001, 20, 'unet_depth3_camus_novgg', 'data_camus1'],
        [UNet, {'depth':4, 'input_size':INPUT_SIZE}, 0.0001, 20, 'unet_depth4_camus_novgg', 'data_camus1'],
        [UNet, {'depth':5, 'input_size':INPUT_SIZE}, 0.0001, 20, 'unet_depth5_camus_novgg', 'data_camus1'],
        
        #[UNETR, {'depth':2, 'skip_connections':[0,1], 'pretrained_name': 'vit_base_patch16_224', 'num_classes':NUM_CLASSES}, 0.0001, 20, 'unetr_depth2_sc01_pre_lr', 'data_rgb_b_noise_len_10k'],
        #[UNETR, {'depth':3, 'skip_connections':[0,1,2], 'pretrained_name': 'vit_base_patch16_224', 'num_classes':NUM_CLASSES}, 0.0001, 20, 'unetr_depth3_camus_sc012', 'data_camus1'],

    ]

    pre_training_parameters = [
        #['unet_vgg_depth5_pre_e5', 50, 0.0001, 'data_len_1000', lambda model: freeze_smp(model, 5), 'unet_vgg_depth5_pre_e55_len1000_b'],
    ]

    for p in parameters:
        cross_validation(*p)

    for p in pre_training_parameters:
        cross_validation_with_pre_training(*p)
    
    '''['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=True), 'unet_frozen_nobottle_2'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=True), 'unet_frozen_nobottle_3'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=True), 'unet_frozen_nobottle_4'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [2,3,4], freeze_bottleneck=False), 'unet_frozen_1'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,3,4], freeze_bottleneck=False), 'unet_frozen_2'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,2,4], freeze_bottleneck=False), 'unet_frozen_3'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,2,3], freeze_bottleneck=False), 'unet_frozen_4'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [], freeze_bottleneck=False), 'unet_frozen_all'],
        ['unet_pre1', 25, 0.001, 'data_b_len_1000', lambda model: UNet_freeze(model, [1,2,3,4], freeze_bottleneck=False), 'unet_frozen_bottle'],'''