'''Evaluate a model on a dataset and save the metrics in a file.'''

from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
from CamusEDImageDataset1 import CamusEDImageDataset
from torch.utils.data import DataLoader
from metrics import MetricsManager
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import segmentation_models_pytorch as smp


# CONSTANTS
MODEL_PATH = 'models/'
DATASET_PATH = 'datasets/'
RESULT_PATH = 'results/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''If true, save a prediction on a sample image after evaluation'''
SAVE_PREDICTION = True

'''Activate evaluation or not'''
METRICS = True

'''Activate receptive field computation or not <!> Only works for UNet class from models.py <!>'''
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
    '''Plot a sample image, its ground truth and the prediction of the model'''
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


def plot_receptive_field(model, dataset, fig):
    '''Plot the receptive field of the model on a sample image'''
    sample_image, sample_mask = dataset[0]
    size = model.get_receptive_field(dilation=kwargs['dilation'])
    width, height = size//2, size//2
    fig.imshow(sample_image.permute(1, 2, 0)[:,:,0], cmap = 'gray')
    # Select the center of receptive field (center on a foreground object)
    y_pos, x_pos = torch.argmax(sample_mask) // 224 + 9, torch.argmax(sample_mask) % 224 + 12
    fig.plot(x_pos, y_pos, 'ro')
    
    # Plot a square around the receptive field and color it in red
    fig.plot([x_pos - width, x_pos + width], [y_pos - height, y_pos - height], 'r')
    fig.plot([x_pos - width, x_pos + width], [y_pos + height, y_pos + height], 'r')
    fig.plot([x_pos - width, x_pos - width], [y_pos - height, y_pos + height], 'r')
    fig.plot([x_pos + width, x_pos + width], [y_pos - height, y_pos + height], 'r')
    fig.fill([x_pos - width, x_pos + width, x_pos + width, x_pos - width], [y_pos - height, y_pos - height, y_pos + height, y_pos + height], 'r', alpha=0.3)
    
    fig.set_xlim(0, 224)
    fig.set_ylim(224, 0)
    fig.axis('off')
    


def save_metrics(metrics_mean, metrics_var, model_name, receptive_field=None):
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
    if receptive_field is not None:
        file.write(f'Receptive field : {receptive_field}\n')
    file.close()



if __name__ == "__main__":
    # List of the models to evaluate by filename (without extension) 
    # The dataset used for training is automatically loaded as it is saved in the model file
    model_names = [
        'unetr_depth4_np',
        'unetr_depth4_sc0123'
    ]

    
    
    
    # EVALUATION LOOP
    rf_fig, fig_ax = plt.subplots(figsize=(25, 20), nrows=4, ncols=5)
    fig_ax = fig_ax.flatten()
    for i, model_name in enumerate(model_names):
        print(f'Evaluating {model_name} ({i+1}/{len(model_names)})')
        
        # LOADING MODEL AND DATA FROM FILES
        checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
        num_classes = checkpoint['num_classes']
        kwargs = checkpoint['kwargs']
        
        dataset_name = checkpoint['datasets']
        dataset = torch.load(DATASET_PATH + 'valid_' +dataset_name + '.pt')
        
        model = checkpoint['model_class'](**kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=DEVICE)
        
        mean_metrics, var_metrics = evaluate(model=model, dataset=dataset)
        
        def model_fn() -> nn.Module:
            model = UNet(**kwargs)
            model.eval()
            return model
        if RECEPTIVE_FIELD:
            rf = model.get_receptive_field(dilation=kwargs['dilation'])
          
        if METRICS:
            if RECEPTIVE_FIELD:
                save_metrics(mean_metrics, var_metrics, model_name, rf)
            else:
                save_metrics(mean_metrics, var_metrics, model_name)
                
        if SAVE_PREDICTION:
            figure = plt.figure(figsize=(10, 5))
            plot_prediction(model=model, dataset=dataset, fig=figure)
            plt.savefig(RESULT_PATH + f'{model_name}_pred.png', bbox_inches='tight')
            if RECEPTIVE_FIELD:
                plot_receptive_field(model=model, dataset=dataset, fig=fig_ax[i])
    
    if RECEPTIVE_FIELD:  
        rf_fig.savefig(RESULT_PATH + f'{model_name}_rf.png', bbox_inches='tight')
        plt.show()

        
    

