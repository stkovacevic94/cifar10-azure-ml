from model import ImageClassifier, NeuralNetwork
from data import CIFAR10DataModule
from logger import ImagePredictionLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
import argparse
import os

def main(hparams):
    deterministic = False
    if hparams.seed is not None:
        seed_everything(hparams.seed, workers=True)
        deterministic = True

    datamodule = CIFAR10DataModule(hparams.data_path, hparams.batch_size)
    
    model = NeuralNetwork()
    system = ImageClassifier(hparams.lr, model)
    
    os.makedirs("./logs", exist_ok=True)
    wandb_logger = WandbLogger(group="Custom CNN", config=hparams, job_type='train', save_dir='./logs')
    wandb_logger.watch(model, log='all')
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        filename='cifar10-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    trainer = Trainer(
            gpus=1, 
            fast_dev_run=False, 
            logger=wandb_logger, 
            max_epochs=hparams.max_epochs,
            track_grad_norm=2,
            weights_summary='full',
            callbacks=[checkpoint_callback],
            #callbacks=[ImagePredictionLogger(val_samples)],
            deterministic=deterministic)

    trainer.fit(system, datamodule)
    wandb.finish()

def add_training_specific_args(parent_parser: argparse.ArgumentParser):
    parser = parent_parser.add_argument_group("Training")
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train')
    return parent_parser

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--seed', type=int, default=None, help='Specific seed for reproducibility')
    parser.add_argument('--logdir', type=str, default="./logs", help='Root directory path for logs')
    # TRAINER level args
    parser = add_training_specific_args(parser)
    # MODEL level args
    parser = ImageClassifier.add_model_specific_args(parser)
    # DATA level args
    parser = CIFAR10DataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)