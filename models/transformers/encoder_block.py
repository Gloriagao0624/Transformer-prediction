import torch
import torch.nn as nn

from .attention import MultiHeadedAttention, TimeAwareMultiHeadAttention
from .utils import SublayerConnection, PointWiseFeedForwardByConv, PointWiseFeedForwardByLinear


class SASTransformerEncoderBlock(nn.Module):
    """
    SAS Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """
    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)

    def forward(self, embed, attn_mask=None, padding_mask=None):

        Q = self.attention_layernorm(embed)
        K, V = embed, embed
        mha_outputs = self.attention_layer(Q, K, V, attn_mask=attn_mask, padding_mask=padding_mask)
        embed = Q + mha_outputs
        embed = self.forward_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = pff_output + embed

        return embed


class TiSASTransformerEncoderBlock(nn.Module):
    """
    TiSAS Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item Embed
            time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed 
            time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed 
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """
    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.attention_layer = TimeAwareMultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)

    def forward(self, embed, time_matrix_K, time_matrix_V, attn_mask=None, padding_mask=None):

        Q = self.attention_layernorm(embed)
        K, V = embed, embed
        mha_outputs = self.attention_layer(Q,
                                           K,
                                           V,
                                           time_matrix_K,
                                           time_matrix_V,
                                           attn_mask=attn_mask,
                                           padding_mask=padding_mask)
        embed = Q + mha_outputs
        embed = self.forward_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = pff_output + embed

        return embed


class BertEncoderBlock(nn.Module):
    """
    Bert Transformer 一层 encoder 的实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """
    def __init__(self, n_heads, d_model, dropout_p=0.1):
        super().__init__()

        self.attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.feed_forward = PointWiseFeedForwardByLinear(d_model=d_model, dropout_p=dropout_p)
        self.input_sublayer = SublayerConnection(size=d_model, dropout_p=dropout_p)
        self.output_sublayer = SublayerConnection(size=d_model, dropout_p=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, embed, attn_mask=None, padding_mask=None):
        embed = self.input_sublayer(
            embed, lambda _x: self.attention.forward(_x, _x, _x, attn_mask=attn_mask, padding_mask=padding_mask))
        embed = self.output_sublayer(embed, self.feed_forward)
        embed = self.dropout(embed)

        return embed


class TransformerEncoderBlock(nn.Module):
    """
    Attention is all you need Transformer 一层 encoder 的标准实现

    Args:
        __init__():
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例, Defaults to 0.1.

        forward():
            embed (Tensor [batch size, series len, d_model]): 输入 item embed
            attn_mask (Tensor [series len, series len]): attention mask 防止因果矛盾：1--mask，0--有效item
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask：1--mask，0--有效item
    Returns:
        forward():
            [Tensor]: [batch size, series len, d_model]
    """
    def __init__(self, n_heads, d_model, dropout_p=0.1):

        super().__init__()
        self.attention_layer = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p)
        self.attention_layernorm = nn.LayerNorm(d_model, eps=1e-8)
        self.forward_layer = PointWiseFeedForwardByLinear(d_model=d_model, d_ff=d_model * 4, dropout_p=dropout_p)
        self.forward_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, embed, attn_mask=None, padding_mask=None):
        Q, K, V = embed, embed, embed
        mha_outputs = self.attention_layer(Q, K, V, attn_mask=attn_mask, padding_mask=padding_mask)
        embed = V + mha_outputs
        embed = self.attention_layernorm(embed)
        pff_output = self.forward_layer(embed)
        embed = embed + pff_output
        embed = self.forward_layernorm(embed)

        return embed
