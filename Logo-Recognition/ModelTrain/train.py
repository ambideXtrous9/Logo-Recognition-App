from model import CNNModel
from EfficientNetB0 import EfficientNet
from Xception import XceptionNet
from InceptionV3 import InceptionV3
from MobilenetV2 import MobileNetV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import LogoDataModule
from lightning.pytorch import seed_everything
import config

seed_everything(42, workers=True)

import argparse

import mlflow.pytorch
from mlflow import MlflowClient



mlflow.set_tracking_uri(config.tracking_uri)
client = MlflowClient(tracking_uri=config.tracking_uri)


def get_model(model_name, num_classes, lr):
    if model_name == "CNN":
        mlflow.set_experiment("CNN")
        return CNNModel(num_classes=num_classes, lr=lr)
    elif model_name == "MobileNet":
        mlflow.set_experiment("MobileNet")
        return MobileNetV2(num_classes=num_classes, lr=lr)
    elif model_name == "EfficientNet":
        mlflow.set_experiment("EfficientNet")
        return EfficientNet(num_classes=num_classes, lr=lr)
    elif model_name == "Xception":
        mlflow.set_experiment("Xception")
        return XceptionNet(num_classes=num_classes, lr=lr)
    elif model_name == "Inception":
        mlflow.set_experiment("Inception")
        return InceptionV3(num_classes=num_classes, lr=lr)
    else:
        raise ValueError("Unknown model name")

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Select model for training')
parser.add_argument('--model', type=str, required=True, help="Model name: 'CNN', 'EfficientNet', MobileNet,or 'XceptionNet'")
args = parser.parse_args()

# Load your configuration, e.g., num_classes, learning_rate, etc.
num_classes = config.NUM_CLASSES
lr = config.LR

# Dynamically select the model
model = get_model(args.model, num_classes, lr)

# Now you can proceed with training the selected model


checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = args.model,
    save_top_k = 1,
    verbose = True,
    monitor = 'mean_val',
    mode = 'min'
)

data_module = LogoDataModule(data_folder=config.DATA_FOLDER,
                            batch_size=config.BATCH_SIZE,
                            val_split=config.VAL_SPLIT)

data_module.setup()

trainer = pl.Trainer(devices=-1, 
                  accelerator="gpu",
                  check_val_every_n_epoch=5,
                  callbacks=[checkpoint_callback],
                  max_epochs=config.MAX_EPOCHS)


mlflow.pytorch.autolog(checkpoint_monitor='mean_val', checkpoint_mode='min')

with mlflow.start_run() as run:
    trainer.fit(model=model,datamodule=data_module)
