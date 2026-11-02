
import torch
import torch.nn as nn


class DilatedCNN(nn.Module):
    """1D dilated CNN; MFCC coefficients are input channels, time is sequence."""

    def __init__(self, n_mfcc=20, num_classes=5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(
                n_mfcc, 32,
                kernel_size=8,
                dilation=4,
                padding="same"
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(
                32, 48,
                kernel_size=4,
                dilation=2,
                padding="same"
            ),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.5)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x, return_features=False):
        h1 = self.block1(x)
        h2 = self.block2(h1)

        pooled = self.pool(h2).squeeze(-1)
        logits = self.classifier(pooled)

        if return_features:
            return logits, h1, h2

        return logits
