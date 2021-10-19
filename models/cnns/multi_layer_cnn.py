import torch.nn as nn
import torch


class MultiCNN(nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.channels = channels
        assert isinstance(self.channels, list), '[!] 参数 channels 得是一个列表！'
        self.kernel_size = kernel_size
        self.p1 = int((self.kernel_size[0] - 1) / 2)
        self.p2 = int((self.kernel_size[1] - 1) / 2)
        self.features = self._make_layers(self.channels)
        # self._conv()

    def _make_layers(self, channels):
        layers = []
        in_channels = 1
        for c in channels:
            layers += [nn.Conv2d(in_channels, c, kernel_size=self.kernel_size, padding=(self.p1, self.p2)),
                       nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True)]
            in_channels = c
        return nn.Sequential(*layers)

    def forward(self, embed):
        if embed.dim() == 3:
            embed = embed.unsqueeze(1)
        embed = self.features(embed)
        # embed = self.conv1(embed)
        # embed = self.bn1(embed)
        # embed = self.relu1(embed)
        # embed = self.conv2(embed)
        # embed = self.bn2(embed)
        # embed = self.relu2(embed)

        app_embed = self.relu1(self.bn1(self.conv1(embed)))
        app_embed = self.relu1(self.bn1(self.conv1(app_embed)))
        return embed

    def _conv(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu1 = nn.ReLU()

# def test():
#     a = MultiCNN([8, 16, 32, 64], [3, 7])
#     x = torch.randn(128, 100, 64)
#     aa = a(x)


# test()
