from dataset_florian import FashionMNISTDataset
from torch.utils.data import DataLoader
from metrics import MetricsManager
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image

# CONSTANTS

MODEL_PATH = 'models/'
DATASET_PATH = 'datasets/'
RESULT_PATH = 'results/unetr/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''If true, save a prediction on a sample image after evaluation'''
SAVE_PREDICTION = True

'''Activate evaluation or not'''
METRICS = True

'''Activate receptive field computation or not'''
RECEPTIVE_FIELD = False

def evaluate(model, dataset, num_classes=10, batch_size=16, device=DEVICE):
    '''Evaluate a model on a specific dataset'''
    model.eval()

    dataloader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle = False)

    # INITIALIZING METRICS
    metrics_manager = MetricsManager(num_classes=num_classes, device=DEVICE)

    # EVALUATION
    with torch.no_grad():
        for img, mask in tqdm(dataloader):
            img = img.to(device=DEVICE)
            mask = mask.to(device=DEVICE)
            mask_pred = model(img)
            metrics_manager.update(mask_pred.argmax(1), mask)
        accuracy, precision, recall, dice_score = metrics_manager.get_overall_metrics()
        accuracy_var, precision_var, recall_var, dice_score_var = metrics_manager.get_metrics_var()
    
    return (accuracy, precision, recall, dice_score), (accuracy_var, precision_var, recall_var, dice_score_var)

def plot_prediction(model, dataset, fig):
    sample_image, sample_mask = dataset[0]
    sample_image_to_show = sample_image.permute(1, 2, 0)[:,:,0]
    fig.add_subplot(1, 3, 1)
    plt.title('image')
    plt.imshow(sample_image_to_show, cmap = 'gray')
    
    plt.axis('off')
    fig.add_subplot(1, 3, 2)
    plt.title('ground truth')
    plt.imshow(sample_mask, vmin=0, vmax=10, cmap='tab10')
    plt.axis('off')
    fig.add_subplot(1, 3, 3)
    plt.title('prediction')
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.array([sample_image.numpy()])).to(device=DEVICE))
        plt.imshow(prediction.argmax(1).cpu().numpy()[0], vmin=0, vmax=10, cmap='tab10')
        plt.axis('off')

def plot_prediction2(model, dataset):
    sample_image, sample_mask = dataset[0]
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.array([sample_image.numpy()])).to(device=DEVICE))
        plt.imshow(prediction.argmax(1).cpu().numpy()[0], vmin=0, vmax=10, cmap='tab10')
        plt.axis('off')

def plot_metrics(model, dataset, metrics):
    accuracy, precision, recall, dice_score = metrics
    plt.ylim(0, 100)
    plt.ylabel('%')
    plt.xlabel('metrics')
    plt.bar(['precision', 'recall', 'dice_score'], [precision * 100, recall * 100, dice_score * 100], color = ['blue', 'orange', 'green'])
    plt.text(0, precision * 100, str(round(float(precision * 100), ndigits=2)))
    plt.text(1, recall * 100,str(round(float(recall * 100), ndigits=2)))
    plt.text(2, dice_score * 100, str(round(float(dice_score * 100), ndigits=2)))
    

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


if __name__ == "__main__":
    model_names = [
        'unetr_d1_10k',
        'unetr_d2_10k',
        'unetr_d3_10k',
        'unetr_d4_10k',
    ]

    # EVALUATION
    
    for i, model_name in enumerate(model_names):
        print(f'Evaluating {model_name} ({i+1}/{len(model_names)})')
        
        # LOADING MODEL AND DATA FROM FILES
        checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
        num_classes = checkpoint['num_classes']
        kwargs = checkpoint['kwargs']
        
        dataset_name = checkpoint['datasets']
        dataset = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
        
        model = checkpoint['model_class'](**kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=DEVICE)
        
        mean_metrics, var_metrics = evaluate(model=model, dataset=dataset)
        
        if SAVE_PREDICTION:
            plt.figure()
            plot_prediction2(model=model, dataset=dataset)
            plt.savefig(RESULT_PATH + f'{model_name}_pred.png', bbox_inches='tight')
            plt.figure()
            plot_metrics(model=model, dataset=dataset, metrics=mean_metrics)
            plt.savefig(RESULT_PATH + f'{model_name}_metrics.png', bbox_inches='tight')
            figure = plt.figure(figsize=(10, 5))
        
        if METRICS:
            save_metrics(mean_metrics, var_metrics, model_name)
        

        
    

