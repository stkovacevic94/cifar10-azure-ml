import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
import argparse

import pytorch_lightning as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Sequential(
            nn.Linear(64*4*4, 128),
            nn.ReLU())
        self.output = nn.Linear(128, 10)

    def forward(self, X):
        X = self.conv_stack1(X)
        X = self.conv_stack2(X)
        X = self.conv3(X)
        X = torch.flatten(X, 1)
        X = self.fc1(X)
        logits = self.output(X)
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