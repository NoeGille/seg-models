from dataset_florian import FashionMNISTDataset, FashionMNISTDatasetRGB
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from models import UNet, UNETR
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import imageio

# CONSTANTS

DATASET_PATH = 'datasets/'
MODEL_PATH = 'models/'
RESULTS = 'results/attention_maps/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 10
BATCH_SIZE = 16
INPUT_SIZE = (224, 224, 1)


def plot_attention_map(block, fig, attention_maps, image = None):
    '''Plot the image, if provided, and all the attention map of th specified block overlayed on it'''
    
    # We get rid of the cls_token
    first_block_attention_maps = attention_maps[0][block][1:, 1:] # dim : (196, 196)
    # PLOTTING THE FIRST ATTENTION MAP
    if image != None:
        plt.imshow(image.permute(1,2,0).cpu().detach().numpy(), cmap='gray', alpha=1)
    for i in range(196):
        fig.add_subplot(14, 14, i+1)
        plt.imshow(first_block_attention_maps[i, :].reshape(14, 14).cpu().detach().numpy(), alpha=0.5)
        plt.axis('off')

def plot_attention_rollout(block, fig, attention_rollout_maps, image=None):
    '''Plot the image, if provided, and all the attention rollout map of the specified block overlayed on it'''

    nth_block_rollout = attention_rollout_maps[block][0][1:, 1:]

    if image != None:
        plt.imshow(image.permute(1,2,0).cpu().detach().numpy(), cmap='gray', alpha=1)
    for j in range(196):
        fig.add_subplot(14, 14, j+1)
        plt.imshow(nth_block_rollout[j, :].reshape(14, 14).cpu().detach().numpy(), alpha=0.5, cmap='jet')
        plt.axis('off')

def plot_rollout_patch(block, patch_idx, attention_rollout_maps, image=None, cls_token=False):
    '''Plot the image, if provided, and the attention rollout map of a specified block and patch overlayed on it.
    If cls_token is True, the rollout of the cls token is plotted'''
    if cls_token:
        nth_block_rollout = attention_rollout_maps[block][0][:, 1:]
        heatmap = nth_block_rollout[0, :].reshape(14, 14).cpu().detach().numpy()
    else:
        nth_block_rollout = attention_rollout_maps[block][0][1:, 1:]
        heatmap = nth_block_rollout[patch_idx, :].reshape(14, 14).cpu().detach().numpy()
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((224, 224))
    heatmap = np.array(heatmap)

    plt.imshow(image.permute(1,2,0).cpu().detach().numpy(), cmap='gray', alpha=0.5)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')

def plot_attention_map_patch(block, patch_idx, attention_maps, image=None):
    '''Plot the image, if provided, and the attention map of a specified block and patch overlayed on it'''
    
    # We get rid of the cls_token
    print(attention_maps.shape)
    first_block_attention_maps = attention_maps[0][block][1:, 1:] # dim : (196, 196)
    heatmap = first_block_attention_maps[patch_idx, :].reshape(14, 14).cpu().detach().numpy()
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((224, 224))
    heatmap = np.array(heatmap)

    plt.imshow(image.permute(1,2,0).cpu().detach().numpy(), cmap='gray', alpha=0.5)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')


def get_object_patch(mask):
    '''Return an array of all foreground objects position and an array of all foreground object patches indexes.
    Returns: int array of objects center of gravity, int array of objects patch indexes
    This function is meant to be used with plot_object_rollout_patch'''
    # SPLITTING THE MASK INTO 4 PARTS
    mask = mask.cpu().detach().numpy()
    mask_1 = mask[:112, :112]
    mask_2 = mask[:112, 112:]
    mask_3 = mask[112:, :112]
    mask_4 = mask[112:, 112:]
    
    # TAKE THE GRAVITY CENTER OF THE OBJECT OF EACH PART
    mask_1 = np.argwhere(mask_1 > 0)
    mask_2 = np.argwhere(mask_2 > 0)
    mask_3 = np.argwhere(mask_3 > 0)
    mask_4 = np.argwhere(mask_4 > 0)

    masks = [mask_1, mask_2, mask_3, mask_4]
    center_masks = []

    for i, mask in enumerate(masks):
        if mask.shape[0] == 0:
            center_masks.append((0, 0))
            continue
        mask_mean = np.mean(mask, axis=0)
        center_mask = (int(mask_mean[0]) + i // 2 * 112, int(mask_mean[1]) + i % 2 * 112)
        center_masks.append(center_mask)
    
    center_masks.remove((0, 0))
    patch_idx = []
    for object_idx in range(len(center_masks)):
        patch_idx.append(center_masks[object_idx][0] // 16 * 14 + center_masks[object_idx][1] // 16)
    return center_masks, patch_idx

def plot_object_rollout_patch(block, object_idx, mask, attention_rollout_maps, image=None):
    '''Plot the image, if provided, and the attention rollout map of one of the object of the image than can be chosen with object_idx argument'''
    center_obj, patch_idx = get_object_patch(mask)
    plot_rollout_patch(block, patch_idx[object_idx], attention_rollout_maps, image)
    plt.plot(center_obj[object_idx][1], center_obj[object_idx][0], c='r', marker='o', markersize=10)

def plot_object_attention_map_patch(block, object_idx, mask, attention_maps, image=None):
    '''Plot the image, if provided, and the attention map of one of the object of the image than can be chosen with object_idx argument'''
    center_obj, patch_idx = get_object_patch(mask)
    plot_attention_map_patch(block, patch_idx[object_idx], attention_maps, image)
    plt.plot(center_obj[object_idx][1], center_obj[object_idx][0], c='r', marker='o', markersize=10)

def save_sample_image(model, sample_image, sample_mask):
    '''Save the sample image, its ground truth and the prediction of the model on it'''
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(sample_image.permute(1,2,0).cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    fig.add_subplot(1, 3, 2)
    plt.title('Ground truth')
    plt.imshow(sample_mask.cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
        
    fig.add_subplot(1, 3, 3)
    plt.title('Prediction')
    pred = model(sample_image.unsqueeze(0).to(device=DEVICE))
    plt.imshow(pred.argmax(1).cpu().detach().numpy()[0], cmap='gray')
    plt.axis('off')
    plt.savefig(RESULTS + model_name + '_sample_pred.png')

# LOAD A UNETR MODEL
'''model_names = ['unetr_depth1_pre_lr_e100', 'unetr_depth6_pre_lr_e100', 'unetr_depth12_pre_lr_e100', 'unetr_depth3_pre_lr_e100',
               'unetr_depth6_pre_lr_e100', 'unetr_depth12_pre_lr_e100', 'unetr_depth3_pre_lr_e100',
               'unetr_depth6_pre_lr_e100', 'unetr_depth12_pre_lr_e100', 
               'unetr_depth12_pre_lr_e100',]

blocks = [0, 0, 0, 0,
          2, 2, 2,
          5, 5,
          11,]'''

model_names = ['unetr_depth3_sc012_pre']
blocks = [2]

has_saved_sample_image = False

# Choose what to do
all_rollout_maps_of_block_i = True
all_attention_maps_of_block_i = True
specific_object_rollout_of_block_i = True
specific_patch_rollout_of_block_i = True
specific_object_attention_map_of_block_i = True
cls_token_of_block_i = False
gif_of_rollout_evolution = False

# If True, use the mean of the attention maps of multiple images instead of image at image_index
multiple_images_rollout_attention_map_mean = False
nb_samples = 100
image_index = 3
object_idx = 1
patch_idx = 132

for i, model_name in enumerate(model_names):
    print(f'Generating attention map for model {model_name} [{i+1}/{len(model_names)}]')
    # LOADING MODEL AND DATA FROM FILES
    checkpoint = torch.load(MODEL_PATH + model_name + '.pt')
    num_classes = checkpoint['num_classes']
    kwargs = checkpoint['kwargs']
    model = checkpoint['model_class'](**kwargs)
    dataset_name = checkpoint['datasets']
    dataset = torch.load(DATASET_PATH + 'valid_' + dataset_name + '.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=DEVICE)
    
    model.eval()
    
    sample_image, sample_mask = dataset[image_index]

    if not has_saved_sample_image:
        save_sample_image(model, sample_image, sample_mask)
        has_saved_sample_image = True
    plt.close()

    if multiple_images_rollout_attention_map_mean:
        samples = [dataset[i] for i in range(nb_samples)]
        images = torch.stack([sample[0] for sample in samples])
        attention_rollout_maps = model.attention_rollout_map(images.to(device=DEVICE), discard_ratio=0.8, head_fusion='max')

        

    attention_rollout_maps = model.attention_rollout_map(sample_image.unsqueeze(0).to(device=DEVICE), discard_ratio=0.8, head_fusion='max')
    attention_maps = model.attention_map(sample_image.unsqueeze(0).to(device=DEVICE))# dim : (n_samples, nb_of_blocks, nb_heads, 197, 197)

    # Plot the rollout map of block i on the sample image
    if all_rollout_maps_of_block_i:
        fig = plt.figure(figsize=(10, 10))
        plt.title(f'Attention rollout map for model {model_name} and block {blocks[i]}')
        plt.axis('off')
        plot_attention_rollout(blocks[i], fig, attention_rollout_maps, sample_image)
        plt.savefig(RESULTS + 'all_attention_rollout/' + model_name + f'_rollout_b{blocks[i]}' + '.png', bbox_inches='tight')
        plt.close()

    if all_attention_maps_of_block_i:
        fig = plt.figure(figsize=(10, 10))
        plt.title(f'Attention map for model {model_name} and block {blocks[i]}')
        plt.axis('off')
        plot_attention_map(blocks[i], fig, attention_maps, sample_image)
        plt.savefig(RESULTS + 'all_attention_maps/' + model_name + f'_attn_b{blocks[i]}' + '.png', bbox_inches='tight')
        plt.close()

    # Plot the rollout of block i of the patch corresponding to the center of an object on the sample image
    if specific_object_rollout_of_block_i:
        

        plt.figure(figsize=(10, 10))
        plt.title(f'Attention map for model {model_name} and block {blocks[i]} on object {object_idx}')
        plt.axis('off')
        plot_object_rollout_patch(blocks[i], object_idx, sample_mask, attention_rollout_maps, sample_image)
        plt.savefig(RESULTS + 'object_attention_rollout/' + model_name + f'_rollout_b{blocks[i]}_obj{object_idx}' + '.png', bbox_inches='tight')
        plt.close()

    if specific_patch_rollout_of_block_i:
        plt.figure(figsize=(10, 10))
        plt.title(f'Attention map for model {model_name} and block {blocks[i]} on patch {patch_idx}')
        plt.axis('off')
        plot_rollout_patch(blocks[i], patch_idx, attention_rollout_maps, sample_image)
        plt.plot(patch_idx % 14 * 16 + 7, patch_idx // 14 * 16 + 7, c='r', marker='o', markersize=10)
        plt.savefig(RESULTS + 'patch_attention_rollout/' + model_name + f'_rollout_b{blocks[i]}_patch{patch_idx}' + '.png', bbox_inches='tight')
        plt.close()

    if specific_object_attention_map_of_block_i:
        plt.figure(figsize=(10, 10))
        plt.title(f'Attention map for model {model_name} and block {blocks[i]} on object {object_idx}')
        plt.axis('off')
        plot_object_attention_map_patch(blocks[i], object_idx, sample_mask, attention_maps, sample_image)
        plt.savefig(RESULTS + 'object_attention_maps/' + model_name + f'_attn_b{blocks[i]}_obj{object_idx}' + '.png', bbox_inches='tight')
        plt.close()

    # Create a gif of the evolution of attention rollout through blocks of a model. 
    # Each frame is the attentioin map of block i of the patch corresponding to the center of an object on the sample image
    if gif_of_rollout_evolution:
        depth = kwargs['depth']
        frames = []
        for t in tqdm(range(depth)):
            plt.figure(figsize=(10, 10))
            plt.title(f'Attention map for model {model_name} and block {t} on object {object_idx}')
            plt.axis('off')
            plot_object_rollout_patch(t, object_idx, sample_mask, attention_rollout_maps, sample_image)
            plt.savefig(RESULTS + 'frame/' + model_name + f'_rollout_b{t}_obj{object_idx}' + '.png', bbox_inches='tight')
            frames.append(imageio.v2.imread(RESULTS + f'frame/' + model_name + f'_rollout_b{t}_obj{object_idx}' + '.png'))
            plt.close()
        imageio.mimsave(RESULTS + model_name + f'_rollout_obj{object_idx}' + '.gif', frames, duration=2000, loop=0)

    if cls_token_of_block_i:
        plt.figure(figsize=(10, 10))
        plt.title(f'Attention map for model {model_name} and block {blocks[i]} on cls token')
        plt.axis('off')
        plot_rollout_patch(blocks[i], 0, attention_rollout_maps, sample_image, cls_token=True)
        plt.savefig(RESULTS + model_name + f'_rollout_b{blocks[i]}_cls_token' + '.png', bbox_inches='tight')
        plt.close()

    print('done')



    #MAKE A GIF OF EVOLUTION OF DISCARD RATIO IN ATTENTION ROLLOUT MAPS
    '''frame = 20

    # GETTING THE ATTENTION ROLLOUT MAPS FROM FIRST BLOCK
    for i in tqdm(range(frame)):
        attention_rollout_maps = model.attention_rollout_map(sample_image.unsqueeze(0).to(device=DEVICE), discard_ratio=(1 * i / frame), head_fusion='mean') # dim : (n_samples, 197, 197)

        # We get rid of the cls_token
        nth_block_rollout = attention_rollout_maps[0][1:, 1:] # dim : (196, 196)

        # PLOTTING THE FIRST ATTENTION ROLLOUT MAP
        fig = plt.figure(figsize=(25, 25))
        plt.imshow(sample_image.permute(1,2,0).cpu().detach().numpy(), cmap='gray', alpha=1)
        for j in range(196):
            fig.add_subplot(14, 14, j+1)
            plt.imshow(nth_block_rollout[j, :].reshape(14, 14).cpu().detach().numpy(), alpha=0.5, cmap='jet')
            plt.axis('off')
        plt.savefig(RESULTS + 'frame_mean/' + model_name + f'_attn_rollout_b_f{i}' + '.png', bbox_inches='tight')
        plt.close()

    frames = []
    for t in tqdm(range(frame)):
        image = imageio.v2.imread(RESULTS + 'frame_mean/' + model_name + f'_attn_rollout_b_f{t}' + '.png')
        frames.append(image)
    imageio.mimsave(RESULTS + model_name + '_attn_rollout_b_mean' + '.gif', frames, duration=500, loop=0)
    print('done')'''