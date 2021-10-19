from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import (Transformer, AbsolutePositionalEmbedding, AppItemEmbedding, TemporalEmbedding)
from .cnns import *

import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(AbstractModel):
    """
    基于Transformer的模型，Embed--transformer--MLP

    Args:
        AbstractModel ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

        self.n_layers = args.sas_num_blocks
        self.n_heads = args.sas_num_heads

        self.app_len = args.series_len - 1
        self.app_num = args.class_num
        self.item_embed_dim = args.item_embed_dim
        self.d_model = args.d_model

        self.task_inputs_dim = args.task_inputs_dim
        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.app_embedding = AppItemEmbedding(app_num=self.app_num, embed_size=self.item_embed_dim)
        self.position_embedding = AbsolutePositionalEmbedding(d_model=self.d_model)
        self.dropout = nn.Dropout(args.dropout_p)

        self.transformer_layer = Transformer(self.n_layers, self.n_heads, self.d_model, args.dropout_p)

        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'embed|tfm|mlp'

    def forward(self, batch):
        _return = {}
        padding_mask = batch['padding_mask']
        embed = self._item_add_feature(batch['app'])

        embed = self.transformer_layer(embed, padding_mask)
        output_logits, y_pred = self._task(embed)

        _return['y_pred'] = y_pred
        _return['output_logits'] = output_logits
        return _return

    def _item_add_feature(self, batch):
        """
        为 item embed 添加 特征 embed

        Args:
            batch (dict): 来自dataloader的一个 batch 的数据

        Returns:
            [Tensor]: [batch size, series len, d_model]
        """
        x = batch['app']
        app_embed = self.app_embedding(x)
        position_embed = self.position_embedding(x)
        embed = self.dropout(app_embed + position_embed)
        return embed

    def _task(self, embed):

        #--- avg pooling = [batch size, d_model] ---#
        embed = F.avg_pool1d(embed.permute(0, 2, 1), embed.size(1)).squeeze(-1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred


class CnnTransformerModel(TransformerModel):
    """
    基于Transformer的模型，Embed--CNN--transformer--MLP

    Args:
        AbstractModel ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)
        self.cnn_layer = CONV_3()

    @classmethod
    def code(cls):
        return 'embed|cnn|tfm|mlp'

    def _item_add_feature(self, batch):
        """
        为 item embed 添加 特征 embed

        Args:
            batch (dict): 来自dataloader的一个 batch 的数据

        Returns:
            [Tensor]: [batch size, series len, d_model]
        """
        x = batch['app']
        app_embed = self.app_embedding(x)
        app_embed = self.cnn_layer(app_embed)
        position_embed = self.position_embedding(x)
        embed = self.dropout(app_embed + position_embed)
        return embed

