import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision 
import os 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels=None, train=True):
        self.imgs = imgs 
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = torchvision.io.read_image(self.imgs[ix]).float() / 255.
        if self.train:
            label = torch.tensor(self.labels[ix], dtype=torch.long)
            return img, label
        return img

class DataModule(pl.LightningDataModule):

    def __init__(self, path, batch_size = 64, test_size = 0.2, seed = 42):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size 
        self.seed = seed 

    def setup(self, stage=None):
        # read csv file with imgs names and labels
        df = pd.read_csv(self.path/'train.csv')
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
        # train dataset
        train_imgs = [str(self.path/'train_images'/img) for img in train['image_id'].values]
        self.train_dataset = Dataset(train_imgs, train['label'].values)
        # val dataset
        val_imgs = [str(self.path/'train_images'/img) for img in val['image_id'].values]
        self.val_dataset = Dataset(val_imgs, val['label'].values)
        # test dataset
        test_imgs = [str(self.path/'test_images'/img) for img in os.listdir(self.path/'test_images')]
        self.test_dataset = Dataset(test_imgs, train=False)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)