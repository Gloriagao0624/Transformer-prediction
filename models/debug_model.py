from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import (Transformer, AbsolutePositionalEmbedding, AppItemEmbedding, LearnedPositionalEmbedding,
                           TemporalEmbedding)
from .cnns import *

import torch.nn as nn
import torch.nn.functional as F


class TimeResAttnModel(AbstractModel):
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

        self.cnn_layer = ResNet_101x64()

        self.app_embedding = AppItemEmbedding(app_num=self.app_num, embed_size=self.item_embed_dim)
        self.time_embedding = TemporalEmbedding(self.d_model)
        self.position_embedding = AbsolutePositionalEmbedding(d_model=self.d_model)
        self.dropout = nn.Dropout(args.dropout_p)

        self.transformer_layer = Transformer(self.n_layers, self.n_heads, self.d_model, args.dropout_p)

        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'super_long_model'

    def forward(self, batch):
        _return = {}
        app_embed = self.app_embedding(batch['app'])
        position_embed = self.position_embedding(batch['app'])
        hour_embed = self.time_embedding(batch)['hour']

        embed = self.dropout(app_embed + position_embed + hour_embed)
        x1_embed, x2_embed, x3_embed = torch.split(embed, [420, 80, 1], dim=1)

        x1_embed = self.cnn_layer(x1_embed)
        x12_embed = torch.cat((x1_embed, x2_embed), dim=1)

        embed = self.transformer_layer(x12_embed)
        embed = torch.cat((embed, x3_embed), dim=1).view(embed.size(0), -1)
        output_logits, y_pred = self._task(embed)

        _return['y_pred'] = y_pred
        _return['output_logits'] = output_logits
        return _return

    def _task(self, embed):
        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)
        return output_logits, y_pred
