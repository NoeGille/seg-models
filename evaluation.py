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


def evaluate(model_name, save_prediction = False, metrics = True):
    '''Evaluate a model on a specified dataset'''
    
    checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
    num_classes = checkpoint['num_classes']
    kwargs = checkpoint['kwargs']
    datasets = checkpoint['datasets']
    
    # LOADING DATASETS
    valid_data = torch.load(DATASET_PATH + 'valid_' + datasets + '.pt')
    test_dataloader = DataLoader(valid_data, batch_size = 16, shuffle = False)

    # LOADING MODEL
    model = checkpoint['model_class'](**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=DEVICE)
    model.eval()

    def model_fn() -> nn.Module:
        model = checkpoint['model_class'](**kwargs)
        model.eval()
        return model

    # INITIALIZING METRICS
    metrics_manager = MetricsManager(num_classes=num_classes, device=DEVICE)

    # EVALUATION
    if metrics:
        with torch.no_grad():
            for img, mask in tqdm(test_dataloader):
                img = img.to(device=DEVICE)
                mask = mask.to(device=DEVICE)
                mask_pred = model(img)
                metrics_manager.update(mask_pred.argmax(1), mask)
            accuracy, precision, recall, dice_score = metrics_manager.get_overall_metrics()
            file = open(RESULT_PATH + f'{model_name}.txt', 'w')
            file.write(f'Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')
            file.write(f'Kwargs : {kwargs}\n')
            file.write(f'Dataset : {datasets}\n')
            file.write(f'Number of epochs : {checkpoint["epochs"]}\n')
            file.write(f'Accuracy : {accuracy:.8f}\n')
            file.write(f'Precision : {precision:.8f}\n')
            file.write(f'Recall : {recall:.8f}\n')
            file.write(f'Dice score : {dice_score:.8f}\n')
            accuracy_var, precision_var, recall_var, dice_score_var = metrics_manager.get_metrics_var()
            file.write(f'Accuracy variance : {accuracy_var:.8f}\n')
            file.write(f'Precision variance : {precision_var:.8f}\n')
            file.write(f'Recall variance : {recall_var:.8f}\n')
            file.write(f'Dice score variance : {dice_score_var:.8f}\n')
            
        if RECEPTIVE_FIELD:
            rf = PytorchReceptiveField(model_fn)
            rf_params = rf.compute(input_shape=checkpoint['input_size'])
            file.write(f'Receptive field : {rf_params}\n')
        file.close()

    # SHOW A PREDICTION
    if save_prediction:
        sample_image, sample_mask = valid_data[0]
        sample_image_to_show = sample_image.permute(1, 2, 0)[:,:,0]
        fig = plt.figure(figsize=(15, 5))
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
        plt.savefig(RESULT_PATH + f'{model_name}_pred.png')
        if RECEPTIVE_FIELD:
            plt.figure()
            rf.plot_rf_grids(
                custom_image=sample_image_to_show,
                figsize=(25, 25),
            )
            plt.savefig(RESULT_PATH + f'{model_name}_rf.png')

if __name__ == "__main__":
    model_names = [
        'unetr_d12',
        'unetr_d12_e100',
        'unetr_d8_sc642',
        'unetr_d4_sc321',
    ]

    # EVALUATION
    for i, model_name in enumerate(model_names):
        print(f'Evaluating {model_name} ({i+1}/{len(model_names)})')
        evaluate(model_name, save_prediction=SAVE_PREDICTION, metrics=METRICS)
    

