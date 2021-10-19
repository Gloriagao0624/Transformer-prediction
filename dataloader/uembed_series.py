from .series import SasSeriesDataLoader, SasSeriesDataset
from .utils import *

import torch
import json
import numpy as np
from tqdm import tqdm


class SasUembedSeriesDataLoader(SasSeriesDataLoader):
    """
    dataloader 返回 app 序列及其 Uembed 序列

    Args:
        SeriesDataLoader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

        # 取得用户 Embed
        self.user_embed_path = args.user_embed_path
        self.user_embed_dict = {}
        with open(self.user_embed_path, 'r') as fh:
            self.user_embed_dict = json.load(fh)

    @classmethod
    def code(cls):
        return 'sas_series|uembed'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = SasUembedSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict,
                                             self.user_embed_dict, self.series_len, self.class_num, self.rng, mode,
                                             self.from_memory)
        elif mode == 'val':
            dataset = SasUembedSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict,
                                             self.user_embed_dict, self.series_len, self.class_num, self.rng, mode,
                                             self.from_memory)
        elif mode == 'test':
            dataset = SasUembedSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict,
                                             self.user_embed_dict, self.series_len, self.class_num, self.rng, mode,
                                             self.from_memory)
        else:
            raise ValueError

        return dataset


class SasUembedSeriesDataset(SasSeriesDataset):
    def __init__(self,
                 dataset_path,
                 user_app_stat,
                 logtail_dict,
                 user_embed_dict,
                 series_len,
                 class_num,
                 rng,
                 mode,
                 from_memory=False):
        super().__init__(dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory)
        self.user_embed_dict = user_embed_dict

    def add_extra_datas(self, data):
        super().add_extra_datas(data)
        uid = data['uid']
        user_embed = self.user_embed_dict[uid]
        self.one_example.update({'user_embed': torch.tensor(user_embed, dtype=torch.float)})
