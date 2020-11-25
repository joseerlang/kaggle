from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import DataModule, Model

config = {
    # optimizer
    'lr': 3e-4,
    'batch_size': 128,
    # training
    'max_epochs': 50,
    'precision': 16,
    'subset': 0.1
}

dm = DataModule(
    path = Path('data'), 
    batch_size=config['batch_size'], 
    subset=config['subset']
)

model = Model(config)

wandb_logger = WandbLogger(project="cassava", config=config)

es = EarlyStopping(monitor='val_acc', mode='max', patience=5)
checkpoint = ModelCheckpoint(dirpath='./', filename='resnet50-{val_acc:.5f}', save_top_k=1, monitor='val_acc', mode='max')

trainer = pl.Trainer(
    gpus=1,
    precision=config['precision'],
    logger=wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint],
)
trainer.fit(model, dm)
