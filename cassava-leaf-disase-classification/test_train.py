from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src import DataModule, Model

path = Path('data')
dm = DataModule(path)

model = Model()

wandb_logger = WandbLogger(project="cassava")

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    limit_train_batches=2,
    limit_val_batches=2,
    logger=wandb_logger,
    max_epochs=5
)
trainer.fit(model, dm)
