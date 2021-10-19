import torch
import torch.nn as nn
from .abstract import AbstractModel
from .dnns import PredictionLayer, MultiLayerPerceptron
from .transformers import AppItemEmbedding


class RnnModel(AbstractModel):
    def __init__(self, args):
        super().__init__(args)
        
        self.item_embed_dim = args.item_embed_dim
        self.app_len = args.series_len - 1
        self.app_num = args.class_num

        self.split_block = args.split_block
        assert self.app_len % self.split_block == 0, '[!] app series 无法被等分为 split_block 块！'
        self.rnn_inputs_dim = int((self.app_len / self.split_block) * self.item_embed_dim)

        self.task_inputs_series_len = args.task_inputs_series_len
        if args.task_inputs_series_len > self.app_len:
            self.task_inputs_dim = self.app_len * self.item_embed_dim + self.item_embed_dim
        else:
            self.task_inputs_dim = self.task_inputs_series_len * self.item_embed_dim + self.item_embed_dim

        self.task_hidden_units = args.task_hidden_units
        self.activation = args.activation
        self.task = args.task

        self.rnn = nn.LSTM(self.rnn_inputs_dim, self.item_embed_dim)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        self.app_embedding = AppItemEmbedding(app_num=self.app_num, embed_size=self.item_embed_dim)
        self.task_layer = MultiLayerPerceptron(self.task_inputs_dim, self.task_hidden_units, self.activation)
        self.predict_layer = PredictionLayer(self.task)

    @classmethod
    def code(cls):
        return 'embed|rnn|mlp'

    def forward(self, batch):
        _return = {}
        embed = self.app_embedding(batch['app'])
        memory = embed[:, -self.task_inputs_series_len:, :].view(embed.size(0), -1)

        embed = torch.chunk(embed, self.split_block, dim=1)
        embed = torch.cat([self.flatten(i.unsqueeze(0)) for i in embed], dim=0)

        _, (ht, _) = self.rnn(embed)
        embed = torch.cat((memory, ht.squeeze(0)), 1)
        output_logits, y_pred = self._task(embed)
        _return['output_logits'] = output_logits
        _return['y_pred'] = y_pred

        return _return

    def _task(self, embed):

        output_logits = self.task_layer(embed)
        y_pred = self.predict_layer(output_logits)

        return output_logits, y_pred
