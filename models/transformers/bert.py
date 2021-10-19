import torch.nn as nn

from .embedding import AbsolutePositionalEmbedding, AppItemEmbedding
from .utils import generate_square_subsequent_mask
from .encoder_block import BertEncoderBlock


class BERT(nn.Module):
    """
    Bert Transformer encoders 的实现

    Args:
        __init__():
            n_layers (int): encoder layers 层数
            n_heads (int): Number of heads for multi-attention
            hidden_dim (int): 隐向量的大小 (d_model)
            dropout_p ([type]): dropout 失活比例

        forward() and predict():
            x_embed (Tensor [batch size, series len, d_model]): 输入 item embed
            pos_embed (Tensor [batch size, series len, d_model]): item embed 对于的正label embed
            neg_embed (Tensor [batch size, series len, d_model]): item embed 对于的负label embed
            padding_mask (Tensor [batch size, series len]): item 序列的padding mask ：1--padding item，0--有效item

    Returns:
        forward():
            pos_logits (Tensor [batch size, series len])
            neg_logits (Tensor [batch size, series len])
            embed (Tensor [batch size, series len, d_model])
        predict()：
            embed (Tensor [batch size, series len, d_model]):
    """

    def __init__(self, n_layers, n_heads, hidden_dim, dropout_p):
        super().__init__()