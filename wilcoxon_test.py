'''Perform a mann-whitney-wilcoxon test on the dice score of the models on their associated dataset.
Plot a table of all the p-values between each models.'''
from dataset_florian import FashionMNISTDatasetRGB
from CamusEDImageDataset1 import CamusEDImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import UNet, UNETR
from scipy.stats import mannwhitneyu
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch


def compute_confusion_matrix(y_pred, y_true):
    '''Compute the confusion matrix for the batch
    Return a tuple : (tp, fp, fn, tn)'''
    tp_k = []
    fp_k = []
    fn_k = []
    tn_k = []
    for k in range(1, 10):
        tp = ((y_pred == k) & (y_true == k)).sum().item()
        fn = ((y_true == k)).sum().item() - tp
        fp = ((y_pred == k)).sum().item() - tp
        tn = len(y_pred.flatten()) - tp - fn - fp
        tp_k.append(tp)
        fp_k.append(fp)
        fn_k.append(fn)
        tn_k.append(tn)
    return torch.mean(torch.tensor(tp_k, dtype=torch.float).to(DEVICE)), torch.mean(torch.tensor(fp_k, dtype=torch.float).to(DEVICE)), torch.mean(torch.tensor(fn_k, dtype=torch.float).to(DEVICE)), torch.mean(torch.tensor(tn_k, dtype=torch.float).to(DEVICE))

def compute_dice_score(y_pred, y_true):
    '''Compute the dice score for the batch'''
    # Avoid division by zero
    tp, fp, fn, tn = compute_confusion_matrix(y_pred, y_true)
    epsilon = 1e-10
    dice_score = 2 * tp / ((2 * tp + fp + fn) + epsilon)
    return dice_score

def evaluate_dice_list(model_name):
    '''evaluate a model and return a list of dice score for each samples of the associated dataset'''
    checkpoint = torch.load('models/' + model_name + '.pt')
    model_class = checkpoint['model_class']
    kwargs = checkpoint['kwargs']
    model = model_class(**kwargs)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    model.eval()

    dataset_name = checkpoint['datasets']

    dataset = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)

    # Evaluate the model
    dice_list = []
    with torch.no_grad():
        for i, (img, mask) in tqdm(enumerate(dataloader), total = len(dataloader)):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            pred = model(img)
            pred = pred.argmax(dim = 1)
            dice = compute_dice_score(pred, mask)
            dice_list.append(dice.mean().item())

    return dice_list

def get_dice_list_from_files(model_name):
    '''Get the dice score list from a file'''
    with open(RESULT_PATH + model_name + '_dice.txt', 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines]

def plot_pvalue_table(model_names, pvalues):
    '''model_names: sequence of length k of the model names.
    pvalues : 2D array k*k of pvalues'''
    # Compute heatmap of colors to highlight the similarity between models
    colors = np.zeros((len(model_names), len(model_names), 3))
    decimals = np.zeros((len(model_names), len(model_names)))
    str_pvalues = np.zeros((len(model_names), len(model_names)), dtype=object)
    # Count number of decimals
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            counter = pvalues[i, j] 
            while counter < 1:
                counter *= 10
                decimals[i, j] += 1
            decimals[i, j] += int(counter) * 0.1
            str_pvalues[i, j] = '{:.1e}'.format(pvalues[i,j])
            print(str_pvalues[i, j])
    decimals = decimals / 10
    decimals = np.where(decimals > 1, 1, decimals)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if i == j:
                colors[i, j] = np.array([0.5, 0.5, 0.5])
            else:
                colors[i, j] = np.array([decimals[i,j], 0.75, 0.1])

    print(pvalues)
    table = plt.table(cellText=str_pvalues, rowLabels=model_names, colLabels=model_names, cellColours=colors, loc='center', cellLoc='center', fontsize=50, bbox=[0, 0, 1, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(7)

def get_pvalues_table_mannwhitney(distribution_list):
    '''Returns a 2D array where each values correspond to a couple i, j pvalue 
    computed with scipy.stats mannwhitney function'''
    pvalues = []
    for dice_list1 in distribution_list:
        pvalues_row = []
        for dice_list2 in distribution_list:
            pvalues_row.append(mannwhitneyu(dice_list1, dice_list2)[1])
        pvalues.append(pvalues_row)
    return np.array(pvalues)
            

def get_distribution_list(model_path, from_files_mode : bool = False):
    '''Get the dice list of each models on each samples of its unique associated dataset.
    Returns a 2D array where each row is a list of the dice of a model on its dataset'''
    distribution_list = []
    for model_name in model_path:
        if from_files_mode:
            dice_list = get_dice_list_from_files(model_name)
        else:
            dice_list = evaluate_dice_list(model_name)
            # Save the results in a file
            with open(RESULT_PATH + model_name + '_dice.txt', 'w') as f:
                for dice in dice_list:
                    f.write(str(dice) + '\n')
        distribution_list.append(dice_list)
        
    return np.array(distribution_list)

    
    

MODEL_PATH = 'models/'
DATASET_PATH = 'datasets/'
RESULT_PATH = 'results/student/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1


# Load the model
model_path = ['unetr_depth1_camus', 'unetr_depth3_camus', 'unetr_depth6_camus']
model_names = ['UNETR depth 1', 'UNETR depth 3', 'UNETR depth 6']

plt.figure(figsize=(10, 10))
plt.axis('off')
distribution_list = get_distribution_list(model_path, from_files_mode = False)
pvalues = get_pvalues_table_mannwhitney(distribution_list)
plot_pvalue_table(np.array(model_names), pvalues)
plt.show()

# The smaller the p-value, the stronger the likelihood that you should reject the null hypothesis.