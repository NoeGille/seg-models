'''Create a new dataset from the FashionMNIST dataset. 
of Mateus Riva, Pietro Gori, Florian Yger, and Isabelle Bloch. Is the u-net directional-relationship aware ?, 2022'''
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
        '''labels : list of labels to include in the foreground objects
        not_labels : list of labels to include in the background objects
        background_obj : number of background objects
        include_label : if True, the label of the foreground objects is included in the mask else the label is 1
        length : number of samples in the dataset
        triangle_mode : if True, the foreground objects are triangles else in square shape
        seed : seed to generate random dataset
        noise : if True, add noise to the background and position of the foreground objects'''
        self.dataset = dataset
        self.transform = transform
        self.shape = shape
        self.labels = labels
        self.not_labels = not_labels
        self.background_obj = background_obj
        self.include_label = include_label
        self.len = length
        self.triangle_mode = triangle_mode
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
        mask = torch.zeros((self.shape, self.shape))#, dtype=torch.uint8)

        try:
            self.noise = self.noise
        except:
            self.noise = False
        
        '''A retirer'''
        #self.noise = False
        '''A retirer'''
        
        # Adding background objects
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
        
        # Select original position for the Cloud of Strutured Objects
        x = random.randint(28, self.shape - 128)
        y = x

        
        if self.noise:
            pos_noise = [[random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)], 
                        [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]]
        else:
            pos_noise = [[0, 0, 0, 0], [0, 0, 0, 0]]
        
        points = [(x, y), (x + 100, y), (x + 100, y + 100), (x, y + 100)]
        
        if self.triangle_mode:
            points = random.sample(points, 3)
        
        # Adding foreground objects
        for i, p in enumerate(points):
            while True:
                img, label = self.random_fashion_mnist(i)
                if label in self.labels:
                    break
            
            x_pos, y_pos = p[0] - 14 + pos_noise[0][i], p[1] - 14 + pos_noise[1][i]
            image[:, x_pos:x_pos+28, y_pos:y_pos+28] = img
            mask[x_pos:x_pos+28, y_pos:y_pos+28] = self.include_label * label + 1
            mask[x_pos:x_pos + 28, y_pos:y_pos + 28][img[0, :, :] == 0] = 0
        
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
        if self.noise:
            std = torch.ones((1, 224, 224)) * 0.05
            mean = torch.zeros((1, 224, 224))
            noise = torch.normal(mean, std)
            image = torch.where(image == 0, torch.abs(noise), image)
            
        return image, mask
    
class FashionMNISTDatasetRGB(FashionMNISTDataset):
    '''A version of the FashionMNISTDataset that returns RGB images where each image is repeated 3 times on the channel axis'''
    def __init__(self, dataset, transform = None, 
                    shape = 224, labels = [1, 2, 3],
                    not_labels = [5, 6, 7], background_obj = 3,
                    include_label = True, length = 10000,
                    triangle_mode = False, seed = -1, noise = False):
        super().__init__(dataset, transform, shape, labels, not_labels, background_obj, include_label, length, triangle_mode, seed, noise)
    
    
    def __getitem__(self, i):
        img, mask = super().__getitem__(i)
        
        # Repeat the image 3 times on the channel axis
        img = img.repeat(3, 1, 1)
        
        return img, mask


if __name__ == "__main__":

    # CONSTANTS

    DATASET_PATH = 'datasets/'
    DATASET_NAME = 'data_hard_b_size10000'

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

    data = FashionMNISTDataset(dataset = train_dataset, transform = valid_transform, length = length, labels = labels, not_labels = not_labels, background_obj = num_background, include_label=False, triangle_mode=True, noise=False)

    print(data[0][0].shape)
    train_data, valid_data = random_split(data, [0.7, 0.3])

    # SAVING DATASETS
    torch.save(data, DATASET_PATH + DATASET_NAME + ".pt")
    
    torch.save(train_data, DATASET_PATH + "train_" + DATASET_NAME + ".pt")
    torch.save(valid_data, DATASET_PATH + "valid_" + DATASET_NAME + ".pt")
    
    # PLOTTING SAMPLES
    if False:
        print()
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        for i in range(1, 10):
            
            image, mask = data[i]
            fig.add_subplot(3, 3, i)
            plt.imshow(image[0, :, :], cmap = 'gray')
            plt.imshow(mask, alpha = 0.5)
            plt.axis('off')
        plt.show()


