import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class ScaledDotProductAttention(nn.Module):
    """
    计算 'Scaled Dot Product Attention'

    Args:
        Q (Tensor [batch size, n head, Q length, head dim]): Target sequence Embed 
        K (Tensor [batch size, n head, K length, head dim]): Src sequence Embed 
        V (Tensor [batch size, n head, V length, head dim]): Src sequence Embed 
            d_model = head dim * n head
        attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None. 1--mask，0--有效item
        padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None. 1--mask，0--有效item
        dropout (Dropout, optional): 传入 dropout 的实例化对象

    Returns:
        (Tensor): [batch size, n heads, Q len, head dim]
        (Tensor): [description]
    """

    def forward(self, Q, K, V, attn_mask=None, padding_mask=None, dropout=None):

        scale = math.sqrt(Q.size(-1))
        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        # energy = [batch size, n head, Q length, K length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale

        if attn_mask is not None:
            
            # [Q len, K len] -> [1, 1, Q len, K len]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            energy += attn_mask

        if padding_mask is not None:
            # [batch size, K len] -> [batch size, 1, 1, K len]
            # padding矩阵中元素为 True 会被 (-2**32+1)极小值 替代
            '''
            这个位置，如果是置换为float('-inf')，会使得softmax的时候出现 nan
            出现原因是，如果padding mask导致一整行都是-inf的时候，无法求softmax
            （-inf,-inf,-inf,-inf）-->softmax-->(nan,nan,nan,nan)
            而 [(-2**32+1)，(-2**32+1)，(-2**32+1)，(-2**32+1)]-->softmax-->[0.25, 0.25, 0.25, 0.25]
            所以置换为一个极小值就好了。比如 (-2**32+1)
            我感觉pytorch的源代码这个位置是有问题的：
            当年第一次写出现nan的时候，百思不得其解，终于找到原因了
            '''
            energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), (-2**32+1))

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        # attention = [batch size, n heads, Q length, K length]
        attention_weight = torch.softmax(energy, dim=-1)

        # p_attn  = [batch size, n heads, Q length, head dim]
        if dropout is not None:
            p_attn = dropout(attention_weight)

        # 第三步，attention结果与V相乘
        x = torch.matmul(p_attn, V)

        return x, attention_weight


class TimeAwareScaledDotProductAttention(nn.Module):
    """
    计算 'Relative time matrix Scaled Dot Product Attention'

    Args:
        Q (Tensor [batch size, n head, Q length, head dim]): Target sequence Embed 
        K (Tensor [batch size, n head, K length, head dim]): Src sequence Embed 
        V (Tensor [batch size, n head, V length, head dim]): Src sequence Embed 
            d_model = head dim * n head
        time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed 
        time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed 
        attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None. 1--mask，0--有效item
        padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None. 1--mask，0--有效item
        dropout (Dropout, optional): 传入 dropout 的实例化对象

    Returns:
        (Tensor): [batch size, n heads, Q len, head dim]
        (Tensor): [description]
    """

    def forward(self, Q, K, V, time_matrix_K, time_matrix_V, attn_mask=None, padding_mask=None, dropout=None):
        scale = math.sqrt(Q.size(-1))

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        # energy = [batch size, n head, Q length, K length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        energy += torch.matmul(time_matrix_K, Q.unsqueeze(-1)).squeeze(-1)

        energy = energy / scale

        if attn_mask is not None:
            # [Q len, K len] -> [1, 1, Q len, K len]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            energy += attn_mask

        if padding_mask is not None:
            # [batch size, K len] -> [batch size, 1, 1, K len]
            # padding矩阵中元素为 True 会被 (-2**32+1)极小值 替代
            '''
            这个位置，如果是置换为float('-inf')，会使得softmax的时候出现 nan
            出现原因是，如果padding mask导致一整行都是-inf的时候，无法求softmax
            （-inf,-inf,-inf,-inf）-->softmax-->(nan,nan,nan,nan)
            而 [(-2**32+1)，(-2**32+1)，(-2**32+1)，(-2**32+1)]-->softmax-->[0.25, 0.25, 0.25, 0.25]
            所以置换为一个极小值就好了。比如 (-2**32+1)
            我感觉pytorch的源代码这个位置是有问题的：
            当年第一次写出现nan的时候，百思不得其解，终于找到原因了
            '''
            energy = energy.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), (-2**32+1))

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        # attention = [batch size, n heads, Q length, K length]
        attention_weight = torch.softmax(energy, dim=-1)

        # p_attn  = [batch size, n heads, Q length, head dim]
        if dropout is not None:
            p_attn = dropout(attention_weight)

        # 第三步，attention结果与V相乘
        x = torch.matmul(p_attn, V)
        x += torch.matmul(attention_weight.unsqueeze(3),
                          time_matrix_V).reshape(x.shape).squeeze(3)

        return x, attention_weight

