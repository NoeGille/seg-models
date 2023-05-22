import albumentations as A
import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision import transforms
import numpy as np
import random

train_dataset = datasets.FashionMNIST(root='./data', train = True, download=True)
valid_dataset = datasets.FashionMNIST(root='./data', train = False, download=True)

class FashionMNISTDataset(Dataset):
    def __init__(self, dataset, transform = None, 
                 shape = 224, labels = [1, 2, 3], 
                 not_labels = [5, 6, 7], background_obj = 3, 
                 include_label = True, length = 10000,
                 triangle_mode = False, seed = -1, noise = False):
        self.dataset = dataset
        self.transform = transform
        self.shape = shape
        self.labels = labels
        self.not_labels = not_labels
        self.background_obj = background_obj
        self.include_label = include_label
        self.len = length
        self.triangle_mode = triangle_mode
        # Use to generate random dataset object
        self.random_key = random.random() if seed < 0 else seed
        self.noise = noise
    
    def random_fashion_mnist(self, i = 0):
        idx = random.randint(0, len(self.dataset) - 1)

        img, label = self.dataset[idx]
        img = transforms.ToTensor()(img)
        
        
        return img, label
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        # Set seed with random_key + index
        random.seed(i + self.random_key)
        
        image = torch.zeros((1, self.shape, self.shape))#, dtype=torch.uint8)
        
        # Add noise to the image
        image = torch.zeros((1, 224, 224))
        

        mask = torch.zeros((self.shape, self.shape))#, dtype=torch.uint8)

        x = random.randint(14, self.shape - 114)
        y = x

        points = [(x, y), (x + 100, y), (x + 100, y + 100), (x, y + 100)]

        if self.triangle_mode:
            points = random.sample(points, 3)
        
        for i, p in enumerate(points):
            while True:
                img, label = self.random_fashion_mnist(i)
                if label in self.labels:
                    break
            
            x_pos, y_pos = p[0] - 14, p[1] - 14
            image[:, x_pos:x_pos+28, y_pos:y_pos+28] = img
            mask[x_pos:x_pos+28, y_pos:y_pos+28] = self.include_label * label + 1
            mask[x_pos:x_pos + 28, y_pos:y_pos + 28][img[0, :, :] == 0] = 0
        
        for i in range(self.background_obj):
            while True:
                img, label = self.random_fashion_mnist(i)
                if label in self.not_labels:
                    break
            while True:
                x = random.randint(0, self.shape - 28)
                y = random.randint(0, self.shape - 28)
                if (mask[x:x+28, y:y+28]).sum().item() == 0:
                    break
            image[:, x:x + 28, y:y + 28] = torch.where(image[:, x:x + 28, y:y + 28] > 0, image[:, x:x + 28, y:y + 28], img)
        
        if self.transform is not None:
            image = image.permute(1, 2, 0)
            image = np.array(image)
            mask = np.array(mask)
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = transforms.ToTensor()(image).to(torch.float)
            mask = torch.from_numpy(mask).long()
        
        # Adding noise to the background
        '''if self.noise:
            std = torch.ones((1, 224, 224)) * 0.05
            mean = torch.zeros((1, 224, 224))
            noise = torch.normal(mean, std)
            image = torch.where(image == 0, torch.abs(noise), image)'''
        return image, mask


if __name__ == "__main__":

    # CONSTANTS

    DATASET_PATH = 'datasets/'
    DATASET_NAME = 'data_hard_b_len10000'

    length = 10000
    num_background = 5
    labels = [0, 1, 2]
    not_labels = [0, 1, 2]

    # TRANSFORMS FUNCTIONS

    p = 0.1
    p_ = 0.2

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p = p),
            A.VerticalFlip(p = p),
            
            A.geometric.rotate.Rotate(limit = 180, p = p_),
        ]
    )

    valid_transform = A.Compose(
        [
            
        ]
    )

    # GENERATING DATASETS

    data = FashionMNISTDataset(dataset = train_dataset, transform = valid_transform, length = length, labels = labels, not_labels = not_labels, background_obj = num_background, include_label=False, triangle_mode=True)

    train_data, valid_data = random_split(data, [0.7, 0.3])

    # SAVING DATASETS
    
    torch.save(train_data, DATASET_PATH + "train_" + DATASET_NAME + ".pt")
    torch.save(valid_data, DATASET_PATH + "valid_" + DATASET_NAME + ".pt")
    
    # PLOTTING SAMPLES
    print()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        
        image, mask = data[i]
        print(image[0, :, :])
        fig.add_subplot(3, 3, i)
        plt.imshow(image[0, :, :], cmap = 'gray')
        plt.axis('off')
    plt.show()
