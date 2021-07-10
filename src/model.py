import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import argparse
import math

import pytorch_lightning as plt

class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockV1, self).__init__()

        out_channels_3x3 = math.floor(out_channels*0.5)
        out_channels_rest = math.ceil(out_channels*0.5)
        out_channels_1x1 = math.floor(out_channels_rest*0.5)
        out_channels_5x5_and_pool = math.ceil(out_channels_rest*0.5)
        out_channels_5x5 = math.floor(out_channels_5x5_and_pool*0.5) 
        out_channels_pool = math.ceil(out_channels_5x5_and_pool*0.5)

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, 1, padding="same"),
            nn.ReLU()
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_3x3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels_3x3, out_channels_3x3, 3, padding="same"),
            nn.ReLU()
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_5x5, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels_5x5, out_channels_5x5, 3, padding="same"),
            nn.ReLU()
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels_pool, 1, padding="same"),
            nn.ReLU()
        )

        self.out_channels = out_channels_1x1 + out_channels_3x3 + out_channels_5x5 + out_channels_pool

    def forward(self, X):
        option_1x1 = self.conv_1x1(X)
        option_3x3 = self.conv_3x3(X)
        option_5x5 = self.conv_5x5(X)
        option_pooling = self.pooling(X)
        return torch.cat((option_1x1, option_3x3, option_5x5, option_pooling), dim=1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        inception1 = InceptionBlockV1(64, 96)
        self.inception1 = nn.Sequential(
            inception1,
            nn.BatchNorm2d(inception1.out_channels),
            nn.Dropout(0.2)
        )
        inception2 = InceptionBlockV1(inception1.out_channels, 128)
        self.inception2 = nn.Sequential(
            inception2,
            nn.BatchNorm2d(inception2.out_channels)
        )

        inception3 = InceptionBlockV1(inception2.out_channels, 256)
        self.inception3 = nn.Sequential(
            inception3,
            nn.BatchNorm2d(inception3.out_channels),
            nn.Dropout(0.5)
        )
        inception4 = InceptionBlockV1(inception3.out_channels, 512)
        self.inception4 = nn.Sequential(
            inception4,
            nn.BatchNorm2d(inception4.out_channels)
        )
        inception5 = InceptionBlockV1(inception4.out_channels, 390)
        self.inception5 = nn.Sequential(
            inception5,
            nn.BatchNorm2d(inception5.out_channels),
            nn.Dropout(0.5)
        )
        inception6 = InceptionBlockV1(inception5.out_channels, 256)
        self.inception6 = nn.Sequential(
            inception6,
            nn.BatchNorm2d(inception6.out_channels)
        )
        inception7 = InceptionBlockV1(inception6.out_channels, 160)
        self.inception7 = nn.Sequential(
            inception7,
            nn.BatchNorm2d(inception7.out_channels)
        )
        inception8 = InceptionBlockV1(inception7.out_channels, 92)
        self.inception8 = nn.Sequential(
            inception8,
            nn.BatchNorm2d(inception8.out_channels)
        )

        self.depth_reducer = nn.Sequential(
            nn.Conv2d(inception8.out_channels, 64, 1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(inception8.out_channels*3*3, 10),
            nn.ReLU(),
        )

        self.aux_conv = nn.Sequential(
            nn.Conv2d(inception3.out_channels, 192, 3),
            nn.ReLU(),
            nn.Conv2d(192, 92, 1),
            nn.ReLU()
        )
        self.aux_fc1 = nn.Sequential(
            nn.Linear(92*2*2, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.aux_out = nn.Linear(128, 10)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = F.max_pool2d(X, 2)
        X = self.inception1(X)
        X = self.inception2(X)
        X = self.inception3(X)

        A = F.avg_pool2d(X, 3)
        A = self.aux_conv(A)
        A = self.aux_fc1(torch.flatten(A, 1))
        aux_logits = self.aux_out(A) 

        X = F.max_pool2d(X, 2)
        X = self.inception4(X)
        X = self.inception5(X)
        X = self.inception6(X)
        X = self.inception7(X)
        X = self.inception8(X)
        X = F.avg_pool2d(X, 2)
        X = torch.flatten(X, 1)
        logits = self.fc(X)
        return logits, aux_logits

class LiteInceptionNetwork(plt.LightningModule):
    def __init__(self, lr, model):
        super().__init__()

        self.model = model

        self.save_hyperparameters("lr")

        self.train_acc = torchmetrics.Accuracy()
        self.train_aux_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.val_aux_acc = torchmetrics.Accuracy()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits, aux_logits = self(X)
        loss = F.cross_entropy(logits, y)
        aux_loss = F.cross_entropy(aux_logits, y)
        self.train_acc(F.softmax(logits, dim=1), y)
        self.train_aux_acc(F.softmax(aux_logits, dim=1), y)
        log_metrics = {
            "train_loss": loss,
            "train_aux_loss": aux_loss, 
            "train_acc": self.train_acc,
            "train_aux_acc": self.train_aux_acc
        }
        self.log_dict(log_metrics, on_step=True, on_epoch=True)
        return 0.5*loss+0.5*aux_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits, aux_logits = self(X)
        loss = F.cross_entropy(logits, y)
        aux_loss = F.cross_entropy(aux_logits, y)
        self.val_acc(F.softmax(logits, dim=1), y)
        self.val_aux_acc(F.softmax(aux_logits, dim=1), y)
        log_metrics = {
            "valid_loss": loss,
            "valid_aux_loss": aux_loss, 
            "valid_acc": self.val_acc,
            "valid_aux_acc": self.val_aux_acc
        }
        self.log_dict(log_metrics)
        return 0.5*loss+0.5*aux_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        return parent_parser