import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import DataModule, Resnet

size = 512
config = {
    # optimization
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 128,
    # data
    'extra_data': 1,
    'subset': 0,
    'test_size': 0.2,
    'seed': 42,
    # model
    'backbone': 'resnet18',
    # data augmentation
    'size': size,
    'train_trans': {
        'PadIfNeeded': {
            'min_height': size,
            'min_width': size,
            'border_mode': 0
        },
        'RandomResizedCrop': {
            'height': size,
            'width': size
        },
        'HorizontalFlip': {},
        'VerticalFlip': {}
    },
    'val_trans': {
        'PadIfNeeded': {
            'min_height': size,
            'min_width': size,
            'border_mode': 0
        },
        'CenterCrop': {
            'height': size,
            'width': size
        }
    },
    # training params
    'precision': 16,
    'max_epochs': 50,
    'val_batches': 1.0
}

dm = DataModule(
    file='train_extra.csv' if config['extra_data'] else 'train_old.csv',
    **config
)

model = Resnet(config)

wandb_logger = WandbLogger(project="cassava", config=config)

es = EarlyStopping(monitor='val_acc', mode='max', patience=3)
checkpoint = ModelCheckpoint(
    dirpath='./', filename=f'{config["backbone"]}-{config["size"]}-da-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')

trainer = pl.Trainer(
    gpus=1,
    precision=config['precision'],
    logger=wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)
