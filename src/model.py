import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedSegCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(DilatedSegCNN, self).__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder with increasing dilation
        self.enc1 = self._conv_block(in_channels, 64, dilation=1)
        self.enc2 = self._conv_block(64, 128, dilation=2)
        self.enc3 = self._conv_block(128, 256, dilation=4)
        self.enc4 = self._conv_block(256, 512, dilation=8)

        # Decoder
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)

        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _conv_block(self, in_c, out_c, dilation):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = x

        # Encoder
        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        # Bottleneck
        x4 = self.enc4(self.pool(x3))

        # Decoder
        x = self.dec3(x4) + x3  # Skip connection
        x = self.dec2(x) + x2
        x = self.dec1(x) + x1

        out = self.classifier(x)
        return out