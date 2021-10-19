from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import (TiSAS, AbsolutePositionalEmbedding, AppItemEmbedding, LearnedPositionalEmbedding,
                           TimeMaskEncoding)
from .cnns import *

import torch.nn as nn
import torch.nn.functional as F


class TiSASModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.n_layers = args.sas_num_blocks
        self.n_heads = args.sas_num_heads

        self.app_len = args.series_len - 1
        self.app_num = args.class_num
        self.item_embed_dim = args.item_embed_dim
        self.d_model = args.d_model

        self.task_inputs_series_len = args.task_inputs_series_len
        if self.task_inputs_series_len > self.app_len:
            self.task_inputs_dim = self.app_len * self.d_model
        else:
            self.task_inputs_dim = self.task_inputs_series_len * self.d_model

        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.app_embedding = AppItemEmbedding(app_num=self.app_num, embed_size=self.item_embed_dim)
        # self.position_embedding = LearnedPositionalEmbedding(d_model=self.d_model, max_len=self.app_len)
        self.position_embedding = AbsolutePositionalEmbedding(d_model=self.d_model)

        # 时间间隔矩阵编码
        self.time_matrix_K_embedding = TimeMaskEncoding(d_model=self.d_model)
        self.time_matrix_V_embedding = TimeMaskEncoding(d_model=self.d_model)

        self.dropout = nn.Dropout(args.dropout_p)

        self.sas_layer = TiSAS(self.n_layers, self.n_heads, self.d_model, args.dropout_p)

        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'embed|tisas|mlp'

    def forward(self, batch):
        _return = {}
        padding_mask = batch['padding_mask']

        embed = self._item_add_feature(batch)
        time_matrix_K = self.time_matrix_K_embedding(batch['time_matrix'][:, :-1, :-1])
        time_matrix_V = self.time_matrix_V_embedding(batch['time_matrix'][:, :-1, :-1])
        if self.training:
            pos_embed = self.app_embedding(batch['pos'])
            neg_embed = self.app_embedding(batch['neg'])
            pos_logits, neg_logits, embed = self.sas_layer(embed, pos_embed, neg_embed, time_matrix_K, time_matrix_V,
                                                           padding_mask)
            _return['pos_logits'] = pos_logits
            _return['neg_logits'] = neg_logits
        else:
            embed = self.sas_layer.predict(embed, time_matrix_K, time_matrix_V, padding_mask)

        output_logits, y_pred = self._task(embed, self.task_inputs_series_len)
        _return['output_logits'] = output_logits
        _return['y_pred'] = y_pred

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

    def _task(self, embed, inputs_series_len):

        #--- concat ---#
        embed = embed[:, -inputs_series_len:, :].squeeze(1)
        embed = embed.view(embed.size(0), -1)

        #--- avg pooling = [batch size, d_model] ---#
        # embed = F.avg_pool1d(embed.permute(0, 2, 1), embed.size(1)).squeeze(-1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred


class CNNTiSASModel(TiSASModel):
    def __init__(self, args):
        super().__init__(args)
        self.cnn_layer = CONV_3()

    @classmethod
    def code(cls):
        return 'embed|cnn|tisas|mlp'

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
