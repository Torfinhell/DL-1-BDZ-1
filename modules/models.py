from .config import Config
from torch import nn
from torchvision.models import resnet50, resnet34, resnet18, mnasnet0_5
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.skip(x)
        out = self.relu(out)
        return self.pool(out)


class MyModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        channels = [3]
        for i in range(config.NUM_BLOCKS):
            channels.append(128 * (2 ** i))

        self.blocks = nn.ModuleList([
            ResidualConvBlock(channels[i], channels[i + 1])
            for i in range(config.NUM_BLOCKS)
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], config.LAST_LINEAR_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)
def get_model(config:Config):
    if(config.MODEL=="MyModel"):
        model=MyModel(config)
    elif(config.MODEL=="RESNET50"):
        model=resnet50(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET34"):
        model=resnet34(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif(config.MODEL=="RESNET18"):
        model=resnet18(weights=None)
        model.fc=nn.Linear(model.fc.in_features,  config.NUM_CLASSES)
    elif config.MODEL == "MNASNET0_5":
        model = mnasnet0_5(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            config.NUM_CLASSES
        )
    else:
        raise NotImplementedError("Model should be implemented")
    return model
