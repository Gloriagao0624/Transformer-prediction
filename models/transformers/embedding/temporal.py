import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class TimeJointEncoding(nn.Module):
    """
    论文：time-dependent representation for neural event sequence prediction。
    对持续时间的编码: Event-time joint Embedding。

    Args:
        d_model (int): 时间编码的隐向量维度
        hidden_embedding_dim ([type]): 
        dropout (float, optional): dropout失活比例. Defaults to 0.1.
    """
    def __init__(self, d_model, hidden_embed_dim, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.hidden_embed_dim = hidden_embed_dim
        self.w = Parameter(torch.Tensor(self.hidden_embed_dim))
        self.b = Parameter(torch.Tensor(self.hidden_embed_dim))
        self.embedding_matrix = Parameter(torch.Tensor(self.hidden_embed_dim, self.d_model))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.embedding_matrix, gain=1.0)
        nn.init.uniform_(self.w)
        nn.init.uniform_(self.b)

    def forward(self, t):
        t = torch.unsqueeze(t, 2).float()  # [batch_size, series_size, 1]
        t = t * self.w + self.b  # [batch_size, series_size, series_size]
        t = F.softmax(t, dim=-1)
        # [batch_size, series_size, d_model]
        output = torch.einsum('bsv,vi->bsi', t, self.embedding_matrix)
        return output


class TimeMaskEncoding(nn.Module):
    """
    # TODO: 会出现 loss Nan
    论文：time-dependent representation for neural event sequence prediction。
    对时间的编码: CONTEXTUALIZING EVENT EMBEDDING WITH TIME MASK。

    Args:
        d_model (int): 时间编码的隐向量维度
        c_dim (int, optional):  Defaults to 32.
        dropout (float, optional): dropout失活比例. Defaults to 0.1.
    """
    def __init__(self, d_model, c_dim=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_dim = c_dim

        self.theta = Parameter(torch.Tensor(self.c_dim, 1))
        self.activation = nn.GELU()

        self.fc = nn.Linear(self.c_dim, self.d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.uniform_(self.theta)

    def forward(self, t):
        # [batch_size, series_size, series_size, 1]
        t = torch.unsqueeze(t, -1).float()
        # [batch_size, series_size, series_size, C]
        t = self.activation(torch.matmul(torch.log(t + 1), self.theta.t()))
        # [batch_size, series_size, series_size, E]
        output = self.sigmoid(self.fc(t))
        return output


class ShiftInvariantTimeEmbedding(nn.Module):
    """
    《Self-attention with functional time representation learning》

    https://github.com/StatsDLMathsRecomSys/Self-attention-with-Functional-Time-Representation-Learning/blob/master/self_attention/modules.py

    Args:
        d_model (int): 时间编码的隐向量维度
    """
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0

        self.effe_numits = d_model // 2
        init_freq = np.linspace(0, 8, self.effe_numits).astype(np.float32)
        # init_freq = 1 / 10 ** init_freq
        ones = np.ones(d_model).astype(np.float32)

        self.cos_w = Parameter(torch.Tensor(init_freq))
        self.sin_w = Parameter(torch.Tensor(init_freq))
        self.beta = Parameter(torch.Tensor(ones))

    def forward(self, t):
        # t.shape = [batch size, temporal length]
        t = torch.unsqueeze(t, 2).repeat(1, 1, self.effe_numits)
        cos_w = 1 / 10**self.cos_w
        sin_w = 1 / 10**self.sin_w
        cos_feat = torch.cos(t * cos_w.view(1, 1, -1))
        sin_feat = torch.sin(t * sin_w.view(1, 1, -1))
        # [N, max_len, d_model]
        freq_feat = torch.cat((cos_feat, sin_feat), -1)
        output = freq_feat * self.beta.view(1, 1, -1)

        # output.shape = [batch size, temporal length, d_model]
        return output


class MercerTimeEmbedding(nn.Module):
    """
    《Self-attention with functional time representation learning》

    https://github.com/StatsDLMathsRecomSys/Self-attention-with-Functional-Time-Representation-Learning/blob/master/self_attention/modules.py

    Args:
        d_model (int): 时间编码的隐向量维度
        time_dim (int, optional): number of dimention for time Embedding. Defaults to None.
        expand_dim (int, optional): degree of frequency expansion. Defaults to 5.
    """
    def __init__(self, d_model, time_dim=None, expand_dim=5):
        super().__init__()

        self.time_dim = d_model if time_dim is None else time_dim

        self.expand_dim = expand_dim

        init_period_base = np.linspace(0, 8, self.time_dim).astype(np.float32)
        self.period_var = Parameter(torch.Tensor(init_period_base))

        self.expand_coef = (torch.arange(0, self.expand_dim) + 1).view(1, -1).type(torch.FloatTensor)

        self.basis_expan_var = Parameter(torch.Tensor(self.time_dim, 2 * self.expand_dim))
        self.basis_expan_var_bias = Parameter(torch.Tensor(self.time_dim))

        nn.init.xavier_normal_(self.basis_expan_var)
        nn.init.zeros_(self.basis_expan_var_bias)

    def forward(self, t):
        # t.shape = [batch size, temporal length]
        t = torch.unsqueeze(t, 2).repeat(1, 1, self.time_dim)

        period_var = 10.0**self.period_var
        period_var = torch.unsqueeze(torch.Tensor(period_var.cpu()).to(t.device), 1).repeat(1, self.expand_dim)

        freq_var = 1 / period_var
        freq_var = freq_var * self.expand_coef.to(t.device)

        sin_enc = torch.sin(torch.unsqueeze(t, -1) * torch.unsqueeze(torch.unsqueeze(freq_var, 0), 0))
        cos_enc = torch.cos(torch.unsqueeze(t, -1) * torch.unsqueeze(torch.unsqueeze(freq_var, 0), 0))
        time_enc = torch.cat((sin_enc, cos_enc), -1) * \
            torch.unsqueeze(torch.unsqueeze(self.basis_expan_var, 0), 0)

        output = torch.sum(time_enc, -1) + \
            torch.unsqueeze(torch.unsqueeze(self.basis_expan_var_bias, 0), 0)

        # output.shape = [batch size, temporal length, d_model]
        return output


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='default'):
        super().__init__()

        hour_size = 25
        mimute_size = 61
        second_size = 61
        weekday_size = 8
        day_size = 32

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)

    def forward(self, batch):

        hour_e = self.hour_embed(batch['hour'])
        week_e = self.weekday_embed(batch['week'])

        ret_time = {'week': week_e, 'hour': hour_e}

        return ret_time
