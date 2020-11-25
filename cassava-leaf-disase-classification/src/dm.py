import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torchvision
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        img = torchvision.io.read_image(self.imgs[ix]).float() / 255.
        label = torch.tensor(self.labels[ix], dtype=torch.long)
        return img, label


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 path=Path('data'),
                 batch_size=64,
                 test_size=0.2,
                 random_state=42,
                 subset=0):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.subset = subset

    def setup(self, stage=None):
        df = pd.read_csv(self.path/'train.csv')
        train, val = train_test_split(
            df,
            test_size=self.test_size,
            shuffle=True,
            stratify=df['label'],
            random_state=self.random_state
        )
        print("Training samples: ", len(train))
        print("Validation samples: ", len(val))
        if self.subset:
            _, subset = train_test_split(
                train,
                test_size=self.subset,
                shuffle=True,
                stratify=train['label'],
                random_state=self.random_state
            )
            print("Training with ", len(subset), " samples.")
            train_imgs = [str(self.path/'train_images'/img)
                          for img in subset['image_id'].values]
            train_labels = subset['label'].values
        else:
            train_imgs = [str(self.path/'train_images'/img)
                          for img in train['image_id'].values]
            train_labels = train['label'].values

        # datasets
        self.train_ds = Dataset(train_imgs, train_labels)

        val_imgs = [str(self.path/'train_images'/img)
                    for img in val['image_id'].values]
        val_labels = val['label'].values
        self.val_ds = Dataset(val_imgs, val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True)
