import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy


class Model(pl.LightningModule):

    def __init__(self, config, n_classes=5):
        super().__init__()
        self.save_hyperparameters(config)
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes)
        self.trans = torch.nn.Sequential(
            torchvision.transforms.CenterCrop(self.hparams.size),
            # ...
        )

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.trans(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.trans(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
