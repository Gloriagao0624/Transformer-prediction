import torch
import torch.nn as nn
import torch.nn.functional as F


class CONV_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        embedding维度每8个卷一次，两层卷为1
        通道两层卷为64
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_1_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        在1的基础上，embedding维度每8个卷一次，跳步减小，卷完每八个做一次max pooling，两层卷为1
        通道两层卷为64
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 8), stride=(1, 2), padding=(2, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=(1, 2), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1, 4))
        self.pool2 = nn.MaxPool2d((1, 4))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_1_2(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        在1的基础上，embedding维度每8个卷一次，跳步大一些，卷完每八个做一次max pooling，两层卷为1
        通道两层卷为64
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 8), stride=(1, 4), padding=(2, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=(1, 4), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1, 2))
        self.pool2 = nn.MaxPool2d((1, 2))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_2(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        embedding维度每8个卷一次，两层卷为1
        通道两层卷为32
        卷积核在时间轴扩大，由5变为15，其余保持一致
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(15, 8), stride=(1, 8), padding=(7, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(15, 8), stride=(1, 8), padding=(7, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_2_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        embedding维度每8个卷一次，两层卷为1
        通道两层卷为32
        在2的基础上，跳步变小，做max pooling
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(15, 8), stride=(1, 2), padding=(7, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(15, 8), stride=(1, 2), padding=(7, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1, 4))
        self.pool2 = nn.MaxPool2d((1, 4))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_3(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        在1的基础上，加深层数，跳步减小
        '''
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=1, padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x

class CONV_3_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 32], 卷积完[B, 32, t, 1]，输出[B, t, 32]
        在1的基础上，加深层数，跳步减小
        '''
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(5, 4), stride=1, padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_4(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 16, t, 4]，输出[B, t, 64]
        在3的基础上，四层，flatten 16*4
        '''
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(5, 16), padding=(2, 0))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 16), padding=(2, 0))
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(5, 16), padding=(2, 0))
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(5, 16), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_5(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 8, t, 8]，输出[B, t, 64]
        在3的基础上，四层，flatten 8*8
        '''
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(5, 15), padding=(2, 0))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 15), padding=(2, 0))
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(5, 15), padding=(2, 0))
        self.conv4 = nn.Conv2d(8, 8, kernel_size=(5, 15), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_6(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        测试avg pooling的效果
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 9), padding=(2, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 9), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 6))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_7(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        测试max pooling的效果
        '''
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 9), padding=(2, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 9), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1, 4))
        self.pool2 = nn.MaxPool2d((1, 6))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class CONV_8(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 8, t, 8]，输出[B, t, 64]
        在1的基础上，两层，flatten 8*8
        '''
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 4), stride=(1, 4), padding=(2, 0))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(5, 2), stride=(1, 2), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_9(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 16, t, 4]，输出[B, t, 64]
        在1的基础上，两层，flatten 16*4
        '''
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 4), stride=(1, 4), padding=(2, 0))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 4), stride=(1, 4), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_10(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 16, t, 4]，输出[B, t, 64]
        # tcn两层，16*4 铺平产开
        # 每层卷积完需要调用self._chomp(output, padding_length)，所有做tcn的都需要按照以下格式进行
        # 例：out = self.relu1(self.bn1(self.conv1(x)))
        #     out = self._chomp(out, 4)
        #     out = self.relu2(self.bn2(self.conv2(out)))
        #     out = self._chomp(out, 4)
        '''
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 17), stride=(1, 2), padding=(4, 0))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 17), stride=(1, 2), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def _chomp(self, x, chomp_size_t):
        return x[:, :, :-chomp_size_t, :]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self._chomp(x, 4)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self._chomp(x, 4)
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_11(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 16, t, 4]，输出[B, t, 64]
        # tcn四层，16*4 铺平产开
        '''
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(5, 16), padding=(4, 0))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 16), padding=(4, 0))
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(5, 16), padding=(4, 0))
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(5, 16), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def _chomp(self, x, chomp_size_t):
        return x[:, :, :-chomp_size_t, :]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self._chomp(x, 4)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self._chomp(x, 4)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self._chomp(x, 4)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self._chomp(x, 4)
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_12(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 16, t, 4]，输出[B, t, 64]
        # tcn六层，16*4 铺平产开
        '''
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(5, 11), padding=(4, 0))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 11), padding=(4, 0))
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(5, 11), padding=(4, 0))
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(5, 11), padding=(4, 0))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(5, 11), padding=(4, 0))
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(5, 11), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def _chomp(self, x, chomp_size_t):
        return x[:, :, :-chomp_size_t, :]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self._chomp(x, 4)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self._chomp(x, 4)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self._chomp(x, 4)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self._chomp(x, 4)
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self._chomp(x, 4)
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self._chomp(x, 4)
        x = self.flatten(x.permute(0, 2, 1, 3).contiguous())
        return x


class CONV_13(nn.Module):
    def __init__(self, c):
        super().__init__()
        '''
        输入[B, c, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        embedding维度每8个卷一次，两层卷为1
        通道两层卷为64
        '''
        self.conv1 = nn.Conv2d(c, 32, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=(1, 8), padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x

class CONV_14(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入[B, t, 64], 卷积完[B, 64, t, 1]，输出[B, t, 64]
        在1的基础上，加深层数，跳步减小
        '''
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5, 8), stride=1, padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return x

def test():
    x = torch.randn([12, 100, 32])
    print(x.shape)
    model = CONV_3_1()
    output = model(x)
    print(output.shape)


# test()