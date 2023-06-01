from evaluation import evaluate
from torch.utils.data import random_split, ConcatDataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from models import UNet, UNETR
from dataset_florian import FashionMNISTDataset

'''Given a model and a dataset, train the model on the dataset and evaluate it
using cross validation'''

DATASET_PATH = 'datasets/cv/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (224, 224, 1)
BATCH_SIZE = 16
NUM_CLASSES = 10
MODEL_PATH = 'models/'
RESULT_PATH = 'results/dilated/'

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
                loss = criterion(mask_pred, mask)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()
                
        mean_metrics, var_metrics = evaluate(model=model, dataset=dataset)
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
        
    
if __name__ == "__main__":
    parameters = [
        [UNet, {'depth':1, 'input_size':INPUT_SIZE, 'dilation':8}, 0.001, 50, 'unet_d1_dil8', 'data_b_len_1000'],
        [UNet, {'depth':2, 'input_size':INPUT_SIZE, 'dilation':8}, 0.001, 50, 'unet_d2_dil8', 'data_b_len_1000'],
        [UNet, {'depth':3, 'input_size':INPUT_SIZE, 'dilation':8}, 0.001, 50, 'unet_d3_dil8', 'data_b_len_1000'],
        [UNet, {'depth':4, 'input_size':INPUT_SIZE, 'dilation':8}, 0.001, 50, 'unet_d4_dil8', 'data_b_len_1000'],
        [UNet, {'depth':5, 'input_size':INPUT_SIZE, 'dilation':8}, 0.001, 50, 'unet_d5_dil8', 'data_b_len_1000'],
        [UNet, {'depth':1, 'input_size':INPUT_SIZE, 'dilation':12}, 0.001, 50, 'unet_d1_dil12', 'data_b_len_1000'],
        [UNet, {'depth':2, 'input_size':INPUT_SIZE, 'dilation':12}, 0.001, 50, 'unet_d2_dil12', 'data_b_len_1000'],
        [UNet, {'depth':3, 'input_size':INPUT_SIZE, 'dilation':12}, 0.001, 50, 'unet_d3_dil12', 'data_b_len_1000'],
        [UNet, {'depth':4, 'input_size':INPUT_SIZE, 'dilation':12}, 0.001, 50, 'unet_d4_dil12', 'data_b_len_1000'],
        [UNet, {'depth':5, 'input_size':INPUT_SIZE, 'dilation':12}, 0.001, 50, 'unet_d5_dil12', 'data_b_len_1000'],
    ]

    
    for p in parameters:
        cross_validation(*p)