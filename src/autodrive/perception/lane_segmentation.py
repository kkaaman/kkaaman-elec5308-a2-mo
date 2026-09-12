import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class LaneUNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.pool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decoder4 = DecoderBlock(
            512,
            256,
            256,
        )

        self.decoder3 = DecoderBlock(
            256,
            128,
            128,
        )

        self.decoder2 = DecoderBlock(
            128,
            64,
            64,
        )

        self.decoder1 = DecoderBlock(
            64,
            64,
            64,
        )

        self.final_up = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.classifier = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        stem = self.stem(x)

        x = self.pool(stem)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        x = self.decoder4(
            enc4,
            enc3,
        )

        x = self.decoder3(
            x,
            enc2,
        )

        x = self.decoder2(
            x,
            enc1,
        )

        x = self.decoder1(
            x,
            stem,
        )

        x = self.final_up(x)

        return self.classifier(x)
