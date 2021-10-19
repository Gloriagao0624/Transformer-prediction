import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()

class Chomp2d(nn.Module):
    def __init__(self, chomp_size_t, chomp_size_e):
        super(Chomp2d, self).__init__()
        self.chomp_size_t = chomp_size_t
        self.chomp_size_e = int(chomp_size_e / 2)

    def forward(self, x):
        '''
        这里默认输入的数据是 [B, c, L, E]
        '''
        return x[:, :, :-self.chomp_size_t, self.chomp_size_e:-self.chomp_size_e]


class TemporalBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, padding_t, padding_e,  dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                           stride=stride, padding=(padding_t, padding_e), dilation=dilation))
        self.chomp1 = Chomp2d(padding_t, padding_e)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size,
                                           stride=stride, padding=(padding_t, padding_e), dilation=dilation))
        self.chomp2 = Chomp2d(padding_t, padding_e)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1)) if in_channel != out_channel else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # x = self.conv1(x)
        # x = self.chomp1(x)
        # x = self.conv2(x)
        # x = self.chomp2(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_channel, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding_t=(kernel_size[0]-1)*dilation_size,
                                     padding_e=(kernel_size[1]-1)*dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, num_channels, kernel_size, dropout=0.0):
        assert kernel_size[1] % 2 == 1, '[!] kernel_size[1] 必须是奇数'
        assert isinstance(self.channels, list), '[!] 参数 channels 得是一个列表！'
        super(TCN, self).__init__()
        input_channel = 1
        self.tcn = TemporalConvNet(input_channel, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = x.unsqueeze(1)      # [B, 1, L, E]
        x = self.tcn(x)
        return x


# def test():
#     model = TCN(input_channel=1, num_channels=[8, 1], kernel_size=(16, 12), dropout=0.0)
#     data = torch.randn((7, 100, 64))
#     print(data.shape)
#     output = model(data)
#     print(output.shape)


# test()
