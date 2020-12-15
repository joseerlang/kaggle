import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import DataModule, Resnet

size = 256
config = {
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 128,
    'max_epochs': 50,
    'precision': 16,
    'subset': 0,
    'test_size': 0.2,
    'seed': 42,
    'size': 256,
    'backbone': 'resnet50',
    'val_batches': 1.0,
    'extra_data': 1,
    'train_trans': {
        'CenterCrop': {
            'height': size, 
            'width': size
        }
    },
    'val_trans': {
        'CenterCrop': {
            'height': size, 
            'width': size
        }
    },
}

dm = DataModule(
    file = 'train_extra.csv' if config['extra_data'] else 'train_old.csv', 
    **config
)

model = Resnet(config)

wandb_logger = WandbLogger(project="cassava", config=config)

es = EarlyStopping(monitor='val_acc', mode='max', patience=3)
checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-{config["size"]}-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')

trainer = pl.Trainer(
    gpus=1,
    precision=config['precision'],
    logger= wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)
