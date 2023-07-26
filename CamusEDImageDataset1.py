'''Load and transforms the Camus dataset in a PyTorch Dataset'''
from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor, RandomRotation
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
import glob
import PIL

class CamusEDImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, indexes=None):
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = glob.glob("./data/database_nifti/**/*_ED.nii.gz")
        self.gt_path = glob.glob("./data/database_nifti/**/*_ED_gt.nii.gz")
        if indexes is not None:
            self.img_path = [self.img_path[i] for i in indexes]
            self.gt_path = [self.gt_path[i] for i in indexes]
        self.images = []
        for path in self.img_path:
            image = nib.load(path)
            image = image.get_fdata()
            image = PIL.Image.fromarray(image).convert("RGB")
            image = np.array(image)
            self.images.append(image)
        self.gts = []
        for path in self.gt_path:
            gt = nib.load(path)
            gt = gt.get_fdata()
            gt = PIL.Image.fromarray(gt).convert("L")
            gt = np.array(gt)
            self.gts.append(gt)
        self.seed = np.random.randint(100000)
            
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = self.images[idx]
        gt_image = self.gts[idx]
        if self.transform:
            torch.random.manual_seed(self.seed + idx)
            image = self.transform(image)
        if self.target_transform:
            torch.random.manual_seed(self.seed + idx)
            gt_image = self.target_transform(gt_image)
        return image, gt_image.squeeze(0)
    

if __name__ == '__main__':
    # Save a dataset in a file with torch.save

    DATASET_LENGTH = 1000

    indexes = set(range(DATASET_LENGTH))
    train_indexes = set(np.random.choice(list(indexes),size=int(DATASET_LENGTH * 0.8),replace=False))
    valid_indexes = indexes - train_indexes

    train_dataset = CamusEDImageDataset(
        transform=Compose([ToPILImage(),Resize((224, 224)),RandomRotation(10),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((224, 224)),RandomRotation(10),PILToTensor()]),
        indexes=train_indexes
    )
    
    valid_dataset = CamusEDImageDataset(
        transform=Compose([ToPILImage(),Resize((224, 224)),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((224, 224)),PILToTensor()]),
        indexes=valid_indexes
    )

    dataset = CamusEDImageDataset(
        transform=Compose([ToPILImage(),Resize((224, 224)),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((224, 224)),PILToTensor()]),
    )

    torch.save(dataset, "../seg-models/datasets/data_camus1.pt")
    torch.save(train_dataset, "../seg-models/datasets/train_data_camus1.pt")
    torch.save(valid_dataset, "../seg-models/datasets/valid_data_camus1.pt")
