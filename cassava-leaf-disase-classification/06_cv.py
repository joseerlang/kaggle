import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from src import DataModule, Resnet
from pathlib import Path
import numpy as np

size = 256
config = {
    # optimization
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 256,
    # data
    'extra_data': 1,
    'upsample': 0,
    'subset': 0.1,
    'test_size': 0.2,
    'seed': 42,
    # model
    'backbone': 'resnet18',
    # data augmentation
    'size': size,
    'train_trans': {
        'RandomCrop': {
            'height': size, 
            'width': size
        },
        'HorizontalFlip': {},
        'VerticalFlip': {}
    },
    'val_trans': {
        'CenterCrop': {
            'height': size, 
            'width': size
        }
    },
    # training params
    'precision': 16,
    'max_epochs': 50,
    'val_batches': 5,
    'folds': 5
}

path = Path('./data')
file = Path('train_extra.csv' if config['extra_data'] else 'train_old.csv')
train = pd.read_csv(path/file)

skf = StratifiedKFold(n_splits=config['folds'], random_state=42, shuffle=True)
for f, (train_index, val_index) in enumerate(skf.split(X=np.zeros(len(train)), y=train['label'])):
    print("Fold: ", f+1)

    dm = DataModule(
        train_data = train.iloc[train_index],
        val_data = train.iloc[val_index], 
        path = path,
        **config
    )

    wandb_logger = WandbLogger(project="cassava", config=config)
    es = EarlyStopping(monitor='val_acc', mode='max', patience=3)
    checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-fold_{f+1}-{{val_acc:.3f}}', save_top_k=1, monitor='val_acc', mode='max')

    model = Resnet(config)

    trainer = pl.Trainer(
        gpus=1,
        precision=config['precision'],
        logger= wandb_logger,
        max_epochs=config['max_epochs'],
        callbacks=[es, checkpoint],
        limit_val_batches=config['val_batches']
    )

    trainer.fit(model, dm)