import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    层归一化之后进行残差连接
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout_p):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))