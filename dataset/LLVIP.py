import os
import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Pad, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAutocontrast, GaussianBlur
from torch import nn
from torch.utils.data import random_split, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
# import matplotlib as plt

class SquarePad(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        (h, w) = img.size
        diff = abs(h - w)

        padding = [diff // 2]
        padding.insert(0 if h < w else -1, 0)
        pad = Pad(padding)

        return pad(img)

class LLVIP(ImageFolder):
    def __init__(self, root, transform=None, patch=False, flip=False, decode=False, classification=False, with_bbox=False):
        super().__init__(root = root, transform = transform)

        self.flip = flip
        self.patch = patch
        self.decode = decode
        self.classification = classification
        self.with_bbox = with_bbox

        self.idx_to_class = {v: float(k) for k, v in self.class_to_idx.items()}


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        (img, num_peoples_idx) = super().__getitem__(index)
        if self.decode: return img, img

        num_peoples = self.idx_to_class[num_peoples_idx]
        
        if not self.classification:
            target = torch.tensor(num_peoples, dtype=torch.float32).unsqueeze(0)
        else:
            target = int(num_peoples)

        if self.with_bbox:
            bbox = np.loadtxt(f'/storage/AN91980/Data/LLVIP/labels/{self.imgs[index][0].split("/")[-1].replace("jpg", "txt")}')
            target = (target, bbox)

        return img, target#, index


# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets

class LLVIPDataset:
    def __init__(self, root = '/data/llvip/infrared/', batch_size=12, subset=False, subset_size=10000, test_subset_size=5000, image_size: tuple = (224, 224), data_augs: list = [], decode = False, classification = False, jpg: bool = True, merge_n_split: bool = True, sub_ratio: float = 1, norm = True, with_bbox = False, shuffle_train=True):
        if not image_size: 
            image_size = (32, 32)

        root = Path(root)
        train_dir, test_dir = "train", "test"
        torch.manual_seed(43)

        p = 0.5
        data_augs = [RandomHorizontalFlip(p)]

        transformations = [
            SquarePad(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        if norm:
            transformations += [transforms.Normalize(0.5, 0.5)]

        dataset = LLVIP(root=root / train_dir, transform=transforms.Compose(transformations), decode=decode, classification=classification, with_bbox=with_bbox)
        train_len = int(round(len(dataset) * 0.85))
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        test_dataset = LLVIP(root=root / test_dir, transform=transforms.Compose(transformations), decode=decode, classification=classification, with_bbox=with_bbox)
        test_len = len(test_dataset)
        self.labels = train_dataset.dataset.classes
        if merge_n_split:
            train_dataset, val_dataset, test_dataset = random_split(ConcatDataset([dataset, test_dataset]), [train_len, val_len, test_len])
            
        self.channels, self.image_size = train_dataset[0][0].shape[-3: -1]

        self.classes = [str(i) for i in range(len(self.labels))]

        dataset_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.dataset_length = dataset_size

        if len(data_augs) > 0:
            transformations_and_data_aug = transformations + data_augs
            train_dataset.transform = transforms.Compose(transformations_and_data_aug)
        self.train_set_length = len(train_dataset) + len(val_dataset)

        if sub_ratio < 1:
            import random
            sub_train_len = int(sub_ratio * train_len)
            train_indices = list(range(train_len))
            random.shuffle(train_indices)
            sub_train_indices = train_indices[:sub_train_len]
            train_dataset = torch.utils.data.Subset(train_dataset, sub_train_indices)
            
            dataset_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
            self.dataset_length = dataset_size

        # TODO: Set num of workers to 0 to get a more meaningful error message
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle_train, num_workers=16, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size, num_workers=16, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size, num_workers=16, pin_memory=True)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def get_dataloaders(self, batch=10):
        return self.train_loader, self.val_loader, self.test_loader

    def __getitem__(self, id):
        return self.dataset.__getitem__(id), id

    def __len__(self):
        return self.dataset_length
    



if __name__ == "__main__":
    data = LLVIPDataset()
    data