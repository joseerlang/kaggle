import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy
import torch

class Model(pl.LightningModule):

    def __init__(self, n_classes=5):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        return self.resnet(x)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer