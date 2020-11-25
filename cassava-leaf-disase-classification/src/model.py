import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy
import torch
from torchvision import transforms


class Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 5)
        self.trans_train = torch.nn.Sequential(
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(
            #    brightness=0.2, 
            #    contrast=0.2, 
            #    saturation=0.1, 
            #    hue=0.1
            #)
        )
        self.trans_test = transforms.CenterCrop(256)

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.trans_train(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.trans_test(x)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = accuracy(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer