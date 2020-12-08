import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision 
import os 
from pathlib import Path
import albumentations as A 
import cv2 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, imgs, labels, trans = None):
        self.path = path
        self.imgs = imgs 
        self.labels = labels
        self.trans = trans 

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = cv2.imread(f'{self.path}/{self.imgs[ix]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.trans:
            img = self.trans(image=img)['image']
        img = torch.from_numpy(img / 255.).float().permute(2,0,1)
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label

class DataModule(pl.LightningDataModule):

    def __init__(self, path='data', file='train_extra.csv', size = 256, batch_size = 64, test_size = 0.2, seed = 42, subset=False, train_trans = None, val_trans=None,**kwargs):
        super().__init__()
        self.path = path
        self.file = file
        self.batch_size = batch_size
        self.test_size = test_size 
        self.seed = seed 
        self.subset = subset
        self.size = size
        self.train_trans = train_trans
        self.val_trans = val_trans

    def setup(self, stage=None):
        # read csv file with imgs names and labels
        df = pd.read_csv(f'{self.path}/{self.file}')
        # split in train / val
        train, val = train_test_split(
            df, 
            test_size=self.test_size,
            shuffle=True, 
            stratify=df['label'],
            random_state = self.seed
        )
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        if self.subset:
            _, train = train_test_split(
                train, 
                test_size=self.subset,
                shuffle=True, 
                stratify=train['label'],
                random_state = self.seed
            )
            print("Training only on ", len(train), " samples")
        # train dataset
        self.train_ds = Dataset(
            self.path,
            train['image_id'].values, 
            train['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else None
        )
        # val dataset
        self.val_ds = Dataset(
            self.path,
            val['image_id'].values, 
            val['label'].values,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else None
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
