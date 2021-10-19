import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LearnedPositionalEmbedding(nn.Embedding):
    """
    带参数的 Position Embedding 层。
    继承一个nn.Embedding，再续上一个dropout。
    因为nn.Embedding中包含了一个可以按索引取向量的权重矩阵weight。
    Args:
        d_model (int): 位置编码的隐向量维度
        max_len (int, optional): 编码允许的最长位置. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__(max_len, d_model)

    def forward(self, x):
        # x.shape = [Batch size, Series Length]
        weight = self.weight.data.unsqueeze(0)  # [max_len,E]->[1,max_len,E]
        x = weight[:, :x.size(1), :].repeat(x.size(0), 1, 1)  # [N, S, E]
        return x


class AbsolutePositionalEmbedding(nn.Module):
    """
    Transformer 位置编码，位置编码与 嵌入 具有相同的d_model维度，因此可以将两者相加。
    该模块没有需要训练的参数

    Args:
        d_model (int): 位置编码的隐向量维度
        max_len (int, optional): 编码允许的最长位置. Defaults to 5000.
        position (Tensor, optional): 位置编码. Defaults to None.
    """

    def __init__(self, d_model, max_len=5000, position=None):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        if position is None:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe = pe.unsqueeze(0).transpose(0, 1) # pytorch自带的transformer需要用这招
        self.register_buffer('pe', pe)  # 将模块添加到持久缓冲区

    def forward(self, x):
        # x.shape = [Batch size, Series Length]
        x = self.pe[:, :x.size(1), :].repeat(x.size(0), 1, 1)
        # x.shape = [batch size, Series Length, d_model]
        return x
