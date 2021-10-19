from .series import SasSeriesDataLoader, SasSeriesDataset
from .utils import *

import torch
import numpy as np


class SasTimeSeriesDataLoader(SasSeriesDataLoader):
    """
    dataloader 返回 app 序列及其 time 特征信息

    Args:
        SeriesDataLoader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return 'sas_series|time'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = SasTimeSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict,
                                           self.series_len, self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'val':
            dataset = SasTimeSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict,
                                           self.series_len, self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'test':
            dataset = SasTimeSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict,
                                           self.series_len, self.class_num, self.rng, mode, self.from_memory)
        else:
            raise ValueError

        return dataset


class SasTimeSeriesDataset(SasSeriesDataset):
    def __init__(self, dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory=False):
        super().__init__(dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory)

    def add_extra_datas(self, data):
        super().add_extra_datas(data)
        hour = data['hour']
        week = data['week']
        duration = data['duration']
        delta_t = data['delta_t'][1:] + data['delta_t'][-1:]
        self.one_example.update({
            'hour':
            torch.tensor(padding1D(hour, self.series_len, 0, np.int32), dtype=torch.long),
            'week':
            torch.tensor(padding1D(week, self.series_len, 0, np.int32), dtype=torch.long),
            'duration':
            torch.tensor(padding1D(duration, self.series_len, 0, np.int32), dtype=torch.long),
            'delta_t':
            torch.tensor(padding1D(delta_t, self.series_len, 0, np.int32), dtype=torch.long),
            # 'time_matrix':
            # torch.tensor(padding2D(compute_time_interval_matrix(struct_time['timestamp_s']), self.series_len, 0,
            #                        np.float), dtype=torch.long),
        })


class Sas2TimeSeriesDataLoader(SasSeriesDataLoader):
    """
    dataloader 返回 app 序列信息 ,Bert 构造方式

    Args:
        SasSeriesDataLoader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return 'sas2_series|time'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = Sas2TimeSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict,
                                            self.series_len, self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'val':
            dataset = Sas2TimeSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict,
                                            self.series_len, self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'test':
            dataset = Sas2TimeSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict,
                                            self.series_len, self.class_num, self.rng, mode, self.from_memory)
        else:
            raise ValueError

        return dataset


class Sas2TimeSeriesDataset(SasSeriesDataset):
    def __init__(self, dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory=False):
        super().__init__(dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory)
        self.dummy_item = 5000

    def train_data(self, hist_list):
        # 将整段历史序列中大于 self.class_num 的类别统一归为 0 类
        raw_hist_list = hist_list.copy()
        hist_list[hist_list >= self.class_num] = 0

        app = np.full([self.series_len], 0, dtype=np.int32)
        pos = np.full([self.series_len], 0, dtype=np.int32)
        neg = np.full([self.series_len], 0, dtype=np.int32)
        padding_mask = np.ones([self.series_len], dtype=np.bool_)

        nxt = hist_list[-1]
        idx = self.series_len - 2

        app[-1] = self.dummy_item
        pos[-1] = nxt
        neg[-1] = random_neg(nxt, self.class_num, self.rng)
        padding_mask[-1] = False

        for i in reversed(hist_list[:-1]):
            app[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neg(nxt, self.class_num, self.rng)
            padding_mask[idx] = False
            nxt = i
            idx -= 1
            if idx == -1:
                break

        self.one_example.update({
            'app': torch.tensor(app, dtype=torch.long),
            'pos': torch.tensor(pos, dtype=torch.long),
            'neg': torch.tensor(neg, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'label': torch.tensor(hist_list[-1], dtype=torch.long),
            'raw_hist_list': torch.tensor(raw_hist_list, dtype=torch.long),
        })

    def val_data(self, hist_list):
        raw_hist_list = hist_list.copy()
        label = hist_list[-1]
        # 将整段历史序列中除了最后一个item 大于 self.class_num 的类别统一归为 0 类
        hist_list[:-1][hist_list[:-1] >= self.class_num] = 0
        hist_list[-1] = self.dummy_item
        app = padding1D(hist_list, self.series_len, 0, np.int32)
        padding_mask = padding1D(np.zeros_like(hist_list), self.series_len, 1, np.bool_)

        self.one_example.update({
            'app': torch.tensor(app, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long),
            'raw_hist_list': torch.tensor(raw_hist_list, dtype=torch.long),
        })

    def test_data(self, hist_list):
        self.val_data(hist_list)

    def add_extra_datas(self, data):
        super().add_extra_datas(data)
        hour = data['hour']
        week = data['week']
        duration = data['duration']
        delta_t = data['delta_t'][1:] + data['delta_t'][-1:]
        self.one_example.update({
            'hour':
            torch.tensor(padding1D(hour, self.series_len, 0, np.int32), dtype=torch.long),
            'week':
            torch.tensor(padding1D(week, self.series_len, 0, np.int32), dtype=torch.long),
            'duration':
            torch.tensor(padding1D(duration, self.series_len, 0, np.int32), dtype=torch.long),
            'delta_t':
            torch.tensor(padding1D(delta_t, self.series_len, 0, np.int32), dtype=torch.long),
            # 'time_matrix':
            # torch.tensor(padding2D(compute_time_interval_matrix(struct_time['timestamp_s']), self.series_len, 0,
            #                        np.float), dtype=torch.long),
        })
