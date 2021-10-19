import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel), nn.ReLU(inplace=True),
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_101x64(nn.Module):
    def __init__(self):
        super().__init__()
        self.inchannel = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.fc = nn.Linear(16, 1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = [B, 1, L, E]
        out = self.conv1(x)  # [B, 16, L, E]
        out = self.layer1(out)  # [B, 32, L/2, E/2]
        out = self.layer2(out)  # [B, 64, L/2/2/2, E/2/2]
        out = self.layer3(out)
        out = self.fc(out)  # [B, 64, L/2/2/2, 1]
        out = out.squeeze(-1).permute((0, 2, 1)).contiguous()  # [B, L/2/2, 64]

        return out

class ResNet_501x64(nn.Module):
    def __init__(self):
        super().__init__()
        self.inchannel = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 3), stride=1, padding=(2, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, kernel_size=3, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, kernel_size=3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, kernel_size=3, stride=4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 4), stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def make_layer(self, block, channels, num_blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, kernel_size, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = [B, 1, L, E]
        out = self.conv1(x)  # [B, 16, L, E]
        out = self.layer1(out)  # [B, 32, L/2, E/2]
        out = self.layer2(out)  # [B, 64, L/2/2/2, E/2/2]
        out = self.layer3(out)
        out = self.conv2(out)
        out = out.squeeze(-1).permute((0, 2, 1)).contiguous()  # [B, L/2/2, 64]

        return out

def test():
    r = ResNet_101x64()
    x = torch.randn((2, 80, 64))
    a = r(x)
