import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import argparse

import pytorch_lightning as plt

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3, out_channels_double_3x3, out_channels_pooling):
        super(InceptionBlock, self).__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, 1, padding="same"),
            nn.ReLU()
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_3x3, 1, padding="same"),
            nn.Relu(),
            nn.Conv2d(out_channels_3x3, out_channels_3x3, 3, padding="same"),
            nn.Relu()
        )
        self.double_conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_double_3x3, 3, padding="same"),
            nn.Relu(),
            nn.Conv2d(out_channels_double_3x3, out_channels_double_3x3, 3, padding="same"),
            nn.Relu()
        )
        self.pooling_3x3 = nn.Sequential(
            nn.MaxPool2d(3, padding="same"),
            nn.Conv2d(in_channels, out_channels_pooling, padding="same"),
            nn.Relu()
        )

        self.out_channels = out_channels_1x1 + out_channels_3x3 + out_channels_double_3x3 + out_channels_pooling

    def forward(self, X):
        option_1x1 = self.conv_1x1(X)
        option_3x3 = self.conv_3x3(X)
        option_5x5 = self.double_conv_3x3(X)
        option_pooling = self.pooling_3x3(X)
        return torch.stack((option_1x1, option_3x3, option_5x5, option_pooling), dim=1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        inception1 = InceptionBlock(32, 16, 16, 16, 16)
        self.inception1 = nn.Sequential(
            inception1,
            nn.BatchNorm2d(inception1.out_channels)
        )
        inception2 = InceptionBlock(inception1.out_channels, 16, 64, 64, 16)
        self.inception2 = nn.Sequential(
            inception2,
            nn.BatchNorm2d(inception2.out_channels)
        )

        inception3 = InceptionBlock(inception2.out_channels, 16, 32, 32, 16)
        self.inception3 = nn.Sequential(
            inception3,
            nn.BatchNorm2d(inception3.out_channels)
        )
        inception4 = InceptionBlock(inception3.out_channels, 16, 16, 16, 16)
        self.inception4 = nn.Sequential(
            inception4,
            nn.BatchNorm2d(inception4.out_channels)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(inception4.out_channels*3*3, 10),
            nn.ReLU(),
            )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = F.max_pool2d(X, 2)
        X = self.inception1(X)
        X = self.inception2(X)
        X = F.max_pool2d(X, 2)
        X = self.inception3(X)
        X = self.inception4(X)
        X = F.avg_pool2d(X, 2)
        X = torch.flatten(X, 1)
        logits = self.fc(X)
        return logits


class ImageClassifier(plt.LightningModule):
    def __init__(self, lr, model):
        super().__init__()

        self.model = model

        self.save_hyperparameters("lr")

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.train_acc(F.softmax(logits, dim=1), y)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.val_acc(F.softmax(logits, dim=1), y)
        self.log_dict({"valid_loss": loss, "valid_acc": self.val_acc})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        return parent_parser