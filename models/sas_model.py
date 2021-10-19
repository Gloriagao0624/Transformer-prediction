from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import (SAS, SAS_NoAttnMask, AbsolutePositionalEmbedding, AppItemEmbedding,
                           LearnedPositionalEmbedding, TemporalEmbedding, TimeJointEncoding)
from .cnns import *

import torch.nn as nn
import torch.nn.functional as F


class SasModel(AbstractModel):
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
        self.dropout = nn.Dropout(args.dropout_p)

        self.sas_layer = SAS(self.n_layers, self.n_heads, self.d_model, args.dropout_p)

        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'embed|sas|mlp'

    def forward(self, batch):
        _return = {}
        padding_mask = batch['padding_mask']
        embed = self._item_add_feature(batch)
        if self.training:
            pos_embed = self.app_embedding(batch['pos'])
            neg_embed = self.app_embedding(batch['neg'])
            pos_logits, neg_logits, embed = self.sas_layer(embed, pos_embed, neg_embed, padding_mask)
            _return["pos_logits"] = pos_logits
            _return['neg_logits'] = neg_logits
        else:
            embed = self.sas_layer.predict(embed, padding_mask)

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


class CnnSasModel(SasModel):
    def __init__(self, args):
        super().__init__(args)
        self.cnn_layer = CONV_3()

    @classmethod
    def code(cls):
        return 'embed|cnn|sas|mlp'

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


class TimeCnnSasModel(SasModel):
    def __init__(self, args):
        super().__init__(args)
        self.sas_layer = SAS_NoAttnMask(self.n_layers, self.n_heads, self.d_model, args.dropout_p)
        self.time_embedding = TemporalEmbedding(self.d_model)
        self.delta_t_embedding = TimeJointEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)
        # self.duration_embedding = TimeJointEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)
        self.cnn_layer = CONV_3()

    @classmethod
    def code(cls):
        return 'embed|cnn|timesas|mlp'

    def _item_add_feature(self, batch):
        """
        为 item embed 添加 特征 embed

        Args:
            batch (dict): 来自dataloader的一个 batch 的数据

        Returns:
            [Tensor]: [batch size, series len, d_model]
        """
        app_embed = self.app_embedding(batch['app'])
        hour_embed = self.time_embedding(batch)['hour']
        deltat_embed = self.delta_t_embedding(batch['delta_t'])
        
        # duration_embed = self.duration_embedding(batch['duration'])
        # d_a, _ = torch.split(duration_embed, [100, 1], dim=1)
        # a_a, a_b = torch.split(app_embed, [100, 1], dim=1)
        # app_embed = torch.cat((a_a + d_a, a_b), dim=1)

        app_embed = self.cnn_layer(app_embed)

        embed = app_embed + hour_embed + deltat_embed
        position_embed = self.position_embedding(batch['app'])
        embed = self.dropout(embed + position_embed)
        # embed = self.cnn_layer(embed)

        return embed


class TimeSasModel(SasModel):
    def __init__(self, args):
        super().__init__(args)
        self.sas_layer = SAS_NoAttnMask(self.n_layers, self.n_heads, self.d_model, args.dropout_p)
        self.time_embedding = TemporalEmbedding(self.d_model)
        self.delta_t_embedding = TimeJointEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)
        # self.duration_embedding = TimeJointEncoding(d_model=self.d_model, hidden_embed_dim=self.d_model)

    @classmethod
    def code(cls):
        return 'embed|timesas|mlp'

    def _item_add_feature(self, batch):
        """
        为 item embed 添加 特征 embed

        Args:
            batch (dict): 来自dataloader的一个 batch 的数据

        Returns:
            [Tensor]: [batch size, series len, d_model]
        """
        app_embed = self.app_embedding(batch['app'])
        hour_embed = self.time_embedding(batch)['hour']
        deltat_embed = self.delta_t_embedding(batch['delta_t'])

        # duration_embed = self.duration_embedding(batch['duration'])
        # d_a, _ = torch.split(duration_embed, [100, 1], dim=1)
        # a_a, a_b = torch.split(app_embed, [100, 1], dim=1)
        # app_embed = torch.cat((a_a + d_a, a_b), dim=1)

        embed = app_embed + hour_embed + deltat_embed

        position_embed = self.position_embedding(batch['app'])
        embed = self.dropout(embed + position_embed)

        return embed


class USasModel(SasModel):
    def __init__(self, args):
        super().__init__(args)
        self.task_inputs_dim += 64
        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)

    @classmethod
    def code(cls):
        return 'embed|usas|mlp'

    def forward(self, batch):
        _return = {}
        padding_mask = batch['padding_mask']
        user_embed = batch['user_embed']
        embed = self._item_add_feature(batch)
        if self.training:
            pos_embed = self.app_embedding(batch['pos'])
            neg_embed = self.app_embedding(batch['neg'])
            pos_logits, neg_logits, embed = self.sas_layer(embed, pos_embed, neg_embed, padding_mask)
            _return["pos_logits"] = pos_logits
            _return['neg_logits'] = neg_logits
        else:
            embed = self.sas_layer.predict(embed, padding_mask)

        output_logits, y_pred = self._task(embed, self.task_inputs_series_len, user_embed)
        _return['output_logits'] = output_logits
        _return['y_pred'] = y_pred
        return _return

    def _task(self, embed, inputs_series_len, user_embed):

        #--- concat ---#
        embed = embed[:, -inputs_series_len:, :].squeeze(1)
        embed = embed.view(embed.size(0), -1)
        embed = torch.cat((embed, user_embed), -1)

        #--- avg pooling = [batch size, d_model] ---#
        # embed = F.avg_pool1d(embed.permute(0, 2, 1), embed.size(1)).squeeze(-1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred
