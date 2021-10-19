import torch
import torch.nn as nn
import numpy as np


class AppItemEmbedding(nn.Embedding):
    """
    App item embedding 层，继承了nn.Embedding类

    Args:
        app_num ([int]): App的个数
        embed_size (int, optional): App Embed 的隐向量维度. Defaults to 64.
        padding_idx ([int], optional): padding_idx是索引值，其索引值对应的位置的embed会被填充为 0. Defaults to None.
        embed_path ([str], optional): embedding 权重文件的路径. Defaults to None.
        requires_train (bool, optional): 是否要求权重参与训练. Defaults to True.
    """

    def __init__(self, app_num, embed_size=64, padding_idx=None, embed_path=None, requires_train=True):
        super().__init__(app_num, embed_size, padding_idx)
        self.embed_path = embed_path
        self.requires_train = requires_train
        if self.embed_path:
            # with open(self.embed_path, 'r') as fn:
            appitem_vector = np.load(self.embed_path)
            self.weight.data.copy_(torch.from_numpy(appitem_vector))

            if not self.requires_train:
                # 如果要固定embedding层参数不参与训练，则切记不能在embedding层用优化器
                self.weight.requires_grad = False

class Embedding(nn.Module):
    def __init__(self, n_token, d_model, embed_path=None, requires_train=True):
        """
        自定义 Embedding 层
        :param embed_path: 嵌入矩阵的文件地址 矩阵被存储成文本文件，一行一个 item 对应的 embedding
        :param requires_train: Embedding参数是否参与训练
        """
        super(Embedding, self).__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.requires_train = requires_train
        self.embedding = nn.Embedding(self.n_token, self.d_model)

        if embed_path:
            self.embed_path = embed_path
            # with open(self.embed_path, 'r') as fn:
            self.embed_vector = np.load(self.embed_path)
            self.embedding.weight.data.copy_(
                torch.from_numpy(self.embed_vector))

            if not self.requires_train:
                # 如果要固定embedding层参数不参与训练，则切记不能在embedding层用优化器
                self.embedding.weight.requires_grad = False

    def forward(self, src):
        # 返回Embedding维度和词表大小
        return self.embedding(src)
