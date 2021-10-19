from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import AppItemEmbedding

import torch.nn as nn
import torch.nn.functional as F


class MLPModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)

        self.item_embed_dim = args.item_embed_dim
        self.app_len = args.series_len - 1
        self.app_num = args.class_num

        self.task_inputs_series_len = args.task_inputs_series_len
        if args.task_inputs_series_len > self.app_len:
            self.task_inputs_dim = self.app_len * self.item_embed_dim
        else:
            self.task_inputs_dim = self.task_inputs_series_len * self.item_embed_dim

        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.app_embedding = AppItemEmbedding(app_num=self.app_num, embed_size=self.item_embed_dim)
        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'embed|mlp'

    def forward(self, batch):
        _return = {}
        embed = self.app_embedding(batch['app'])
        output_logits, y_pred = self._task(embed, self.task_inputs_series_len)
        _return['output_logits'] = output_logits
        _return['y_pred'] = y_pred
        return _return

    def _task(self, embed, inputs_series_len):

        #--- concat ---#
        embed = embed[:, -inputs_series_len:, :]
        embed = embed.view(embed.size(0), -1)

        #--- avg pooling = [batch size, d_model] ---#
        # embed = F.avg_pool1d(embed.permute(0, 2, 1), embed.size(1)).squeeze(-1)

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred
