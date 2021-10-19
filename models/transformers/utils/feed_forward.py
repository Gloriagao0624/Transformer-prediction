import torch.nn as nn


class PointWiseFeedForwardByLinear(nn.Module):
    """
    Attention is all you need Transformer FeedForward 实现
    基于 nn.Linear() 实现 Attention机制中的 Feed Forward 函数

    Args:
        d_model (int): 编码的隐向量维度
        d_ff (int, optional): 隐层的维数，通常 4 * hidden_size . Defaults to None.
        dropout_p (float, optional): dropout失活比例. Defaults to 0.1.
    """
    def __init__(self, d_model, d_ff=None, dropout_p=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.dropout2(self.w_2(self.dropout1(self.activation(self.w_1(x)))))


class PointWiseFeedForwardByConv(nn.Module):
    """
    基于 nn.Conv1d() 实现 Attention机制中的 Feed Forward 函数

    Args:
        d_model (int): 编码的隐向量维度
        dropout_p (float, optional): dropout失活比例. Defaults to 0.1.
    """
    def __init__(self, d_model, dropout_p):
        super().__init__()

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.activation = nn.GELU()  # bert用gelu，原生transformer Relu
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        outputs = self.dropout2(self.conv2(self.activation(self.dropout1(self.conv1(x.transpose(-1, -2))))))
        # Conv1D 输入维度要求 (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        return outputs