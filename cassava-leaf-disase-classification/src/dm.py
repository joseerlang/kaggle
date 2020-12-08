import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision
import os
from pathlib import Path
import math
import cv2 
import albumentations as A 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, imgs, labels, trans=None):
        self.path = path
        self.imgs = imgs
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        #img = torchvision.io.read_image(
        #    f'{self.path}/{self.imgs[ix]}').float() / 255.
        img = cv2.imread(f'{self.path}/{self.imgs[ix]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.trans:
            img = self.trans(image=img)['image']
        img = torch.tensor(img / 255., dtype=torch.float).permute(2,0,1)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label


class DataModule(pl.LightningDataModule):

    def __init__(self, train_data, val_data, path='data', batch_size=64, subset=False, train_trans=None, val_trans=None, **kwargs):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.subset = subset
        self.train_data = train_data
        self.val_data = val_data
        self.train_trans = train_trans
        self.val_trans = val_trans

    def setup(self, stage=None):
        print("Training samples: ", len(self.train_data))
        print("Validation samples: ", len(self.val_data))
        if self.subset:
            _, self.train_data = train_test_split(
                self.train_data,
                test_size=self.subset,
                shuffle=True,
                stratify=self.train_data['label'],
                random_state=42
            )
            print("Training only on ", len(self.train_data), " samples")
        # train dataset
        self.train_ds = Dataset(
            self.path,
            self.train_data['image_id'].values,
            self.train_data['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else None
        )
        # val dataset
        self.val_ds=Dataset(
            self.path,
            self.val_data['image_id'].values,
            self.val_data['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else None
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
