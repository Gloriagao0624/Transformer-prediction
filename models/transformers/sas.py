import torch.nn as nn
from .encoder_block import SASTransformerEncoderBlock, TransformerEncoderBlock

from .utils import generate_square_subsequent_mask


class SAS(nn.Module):
    """
    SAS Transformer encoders 的实现

    Args:
        __init__():
            n_layers (int): encoder layers 层数
            n_heads (int): Number of heads for multi-attention
            d_model (int): 隐向量的大小 (d_model)
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
    def __init__(self, n_layers, n_heads, d_model, dropout_p):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([
            SASTransformerEncoderBlock(n_heads=n_heads, d_model=d_model, dropout_p=dropout_p) for _ in range(n_layers)
        ])
        # 正方形的注意力掩码，防止因果矛盾
        self.attn_mask = None
        # 在完整的Transformer encoder-decoder实现中，返回的memory在经过若干个TransformerEncoderBlock之后
        # 最后还得经过一层LayerNor()， 这是因为memory会作为src-tgt的attention layer 的 K V，但是我们此处只是用了encoder
        # 这种情况下，是否还需要LN呢？？
        self.last_layernorm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, x_embed, pos_embed, neg_embed, padding_mask=None):

        embed = self._calculate(x_embed, padding_mask)

        pos_logits = (embed * pos_embed).sum(dim=-1)
        neg_logits = (embed * neg_embed).sum(dim=-1)

        return pos_logits, neg_logits, embed

    def predict(self, x_embed, padding_mask=None):
        embed = self._calculate(x_embed, padding_mask)
        return embed

    def _calculate(self, embed, padding_mask):
        # attn mask 上三角方阵掩码生成
        # embed.shape = [batch size, serise len, d_model]
        if self.attn_mask is None or self.attn_mask.size(0) != embed.shape[1]:
            # 如果没有定义mark矩阵，则根据embed的尺寸生成它
            mask = generate_square_subsequent_mask(embed.shape[1]).to(embed.device)
            self.attn_mask = mask  # [S, S]

        for encoder_block in self.encoder_blocks:
            embed = encoder_block(embed, self.attn_mask, padding_mask)

        embed = self.last_layernorm(embed)

        return embed


class SAS_NoAttnMask(SAS):
    """
    相比 SAS 这个只是没了 attn mask

    Args:
        SAS ([type]): [description]
    """
    def __init__(self, n_layers, n_heads, d_model, dropout_p):
        super().__init__(n_layers, n_heads, d_model, dropout_p)

    def _calculate(self, embed, padding_mask):
        for encoder_block in self.encoder_blocks:
            embed = encoder_block(embed, padding_mask=padding_mask)
        embed = self.last_layernorm(embed)

        return embed
