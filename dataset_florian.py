import albumentations as A
import torch
from torchvision import transforms
from torch.utils.data import Dataset
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
                 triangle_mode = False):
        self.dataset = dataset
        self.transform = transform
        self.shape = shape
        self.labels = labels
        self.not_labels = not_labels
        self.background_obj = background_obj
        self.include_label = include_label
        self.len = length
        self.triangle_mode = triangle_mode
    
    def random_fashion_mnist(self):
        idx = np.random.randint(0, len(self.dataset))

        img, label = self.dataset[idx]
        img = transforms.ToTensor()(img)
        return img, label
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        
        image = torch.zeros((1, self.shape, self.shape))#, dtype=torch.uint8)
        mask = torch.zeros((self.shape, self.shape))#, dtype=torch.uint8)

        x = random.randint(14, self.shape - 114)
        y = x

        points = [(x, y), (x + 100, y), (x + 100, y + 100), (x, y + 100)]

        if self.triangle_mode:
            points = random.sample(points, 3)
        
        for i in points:
            while True:
                img, label = self.random_fashion_mnist()
                if label in self.labels:
                    break
            
            x_pos, y_pos = i[0] - 14, i[1] - 14
            image[:, x_pos:x_pos+28, y_pos:y_pos+28] = img
            mask[x_pos:x_pos+28, y_pos:y_pos+28] = self.include_label * label + 1
            mask[x_pos:x_pos + 28, y_pos:y_pos + 28][img[0, :, :] == 0] = 0
        
        for i in range(self.background_obj):
            while True:
                img, label = self.random_fashion_mnist()
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
        return image, mask


if __name__ == "__main__":
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

    # Easy version (different labels for background and foreground)
    labels = [0, 1, 2]
    not_labels = [3, 4, 5]
    num_background = 6
    train_len = 6000
    valid_len = 6000

    train_data_easy = FashionMNISTDataset(dataset = train_dataset, transform = valid_transform, length = train_len, labels = labels, not_labels = not_labels, background_obj = num_background, include_label=True, triangle_mode=True)#None)#transform = train_transform)
    valid_data_easy = FashionMNISTDataset(dataset = valid_dataset, transform = valid_transform, length = valid_len, labels = labels, not_labels = not_labels, background_obj = num_background, include_label=True, triangle_mode=True)#transform = None)#valid_transform)
    print(len(train_data_easy), len(valid_data_easy))

    # Hard version (same labels for background and foreground)
    labels = [0, 1, 2]
    not_labels = [0, 1, 2]
    num_background = 5
    train_len = 6000
    valid_len = 6000

    train_data_hard = FashionMNISTDataset(dataset = train_dataset, transform = valid_transform, length = train_len, labels = labels, not_labels = not_labels, background_obj = num_background)#None)#transform = train_transform)
    valid_data_hard = FashionMNISTDataset(dataset = valid_dataset, transform = valid_transform, length = valid_len, labels = labels, not_labels = not_labels, background_obj = num_background)#transform = None)#valid_transform)

    torch.save(train_data_easy, "datasets/train_data_easy.pt")
    torch.save(valid_data_easy, "datasets/valid_data_easy.pt")