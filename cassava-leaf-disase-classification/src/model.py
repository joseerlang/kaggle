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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = accuracy(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)

class Resnet(Model):
    def __init__(self, config):
        super().__init__(config)
        self.resnet = getattr(torchvision.models, self.hparams.backbone)(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 5)
    
    def forward(self, x):
        return self.resnet(x)