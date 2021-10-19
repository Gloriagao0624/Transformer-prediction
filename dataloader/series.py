from .abstract import AbstractDataloader, AbstractDataset
from .utils import *

import torch
import numpy as np
from tqdm import tqdm


class SasSeriesDataLoader(AbstractDataloader):
    """
    dataloader 返回 app 序列信息 ,Sas 构造方式

    Args:
        AbstractDataloader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

        self.longtail_path = args.longtail_path

        self.logtail_dict = {}  # 样本对应的长尾app
        with open(self.longtail_path, 'r') as fh:
            for line in tqdm(fh, desc='[*] 正在载入用户长尾列表 '):
                longtail = line.strip('\n').split('\t')
                self.logtail_dict[longtail[0]] = int(longtail[1].split('\t')[0].split(':')[0]) \
                    if longtail[1] is not None and longtail[1] != '' else 0

        self.user_app_stat = {'train': {}, 'val': {}, 'test': {}}
        self.app_stat_path = {
            'train': args.train_app_stat_path,
            'val': args.val_app_stat_path,
            'test': args.test_app_stat_path
        }

        for name, _path in self.app_stat_path.items():
            cache_dict = {}
            with open(_path, 'r') as fh:
                for line in tqdm(fh, desc='[*] 正在载入 {} 列表及统计信息 '.format(name)):
                    values = line.strip('\n').split('\t')
                    if values[0] is None or values[0] == '':  # 处理杰哥数据集上的错误数据
                        continue
                    cache_list = []
                    for v in values[1].split():
                        v = int(v.split(':')[0])
                        cache_list.append(v if v < self.class_num else 0)
                    cache_dict[values[0]] = cache_list
            self.user_app_stat.update({name: cache_dict})

    @classmethod
    def code(cls):
        return 'sas_series'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = SasSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'val':
            dataset = SasSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'test':
            dataset = SasSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        else:
            raise ValueError

        return dataset


class SasSeriesDataset(AbstractDataset):
    def __init__(self, dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory=False):
        super().__init__(dataset_path, series_len, class_num, rng, mode, from_memory)
        self.logtail_dict = logtail_dict
        self.user_app_stat = user_app_stat
        self.app_len = self.series_len - 1  # 历史序列数据集样本长度-1等于需要制造的样本长度

    def train_data(self, hist_list):
        # 将整段历史序列中大于 self.class_num 的类别统一归为 0 类
        raw_hist_list = hist_list.copy()
        hist_list[hist_list >= self.class_num] = 0

        app = np.full([self.app_len], 0, dtype=np.int32)
        pos = np.full([self.app_len], 0, dtype=np.int32)
        neg = np.full([self.app_len], 0, dtype=np.int32)
        padding_mask = np.ones([self.app_len], dtype=np.bool_)
        nxt = hist_list[-1]
        idx = self.app_len - 1

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
        # 将整段历史序列中除了最后一个item 大于 self.class_num 的类别统一归为 0 类
        hist_list[:-1][hist_list[:-1] >= self.class_num] = 0

        app = padding1D(hist_list[:-1], self.app_len, 0, np.int32)
        padding_mask = padding1D(np.zeros_like(hist_list[:-1]), self.app_len, 1, np.bool_)

        self.one_example.update({
            'app': torch.tensor(app, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'label': torch.tensor(hist_list[-1], dtype=torch.long),
            'raw_hist_list': torch.tensor(raw_hist_list, dtype=torch.long),
        })

    def test_data(self, hist_list):
        self.val_data(hist_list)

    def add_extra_datas(self, data):

        uid = data['uid']
        hist_list = [int(v) for v in data['app']]
        longtail = self.logtail_dict[uid] if uid in self.logtail_dict else 0

        candidate = self.user_app_stat[uid]
        candidate = torch.tensor(candidate, dtype=torch.long)
        candidate_mulit_hot = torch.zeros(self.class_num).scatter_(0, candidate, 1)

        longtail_by_x = list(filter(lambda x: x > 5000, hist_list))
        longtail_by_x = max(longtail_by_x, key=longtail_by_x.count) if len(longtail_by_x) != 0 else longtail

        self.one_example.update({
            'uid': torch.tensor(int(uid), dtype=torch.long),
            'longtail': torch.tensor(longtail, dtype=torch.long),
            'longtail_by_x': torch.tensor(longtail_by_x, dtype=torch.long),
            'candidate': candidate_mulit_hot,
        })


class BertSeriesDataLoader(SasSeriesDataLoader):
    """
    dataloader 返回 app 序列信息 ,Bert 构造方式

    Args:
        SeriesDataLoader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)
        self.dummy_prob = args.dummy_prob

    @classmethod
    def code(cls):
        return 'bert_series'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = BertSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict,
                                        self.series_len, self.class_num, self.rng, mode, self.dummy_prob,
                                        self.from_memory)
        elif mode == 'val':
            dataset = BertSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                        self.class_num, self.rng, mode, self.dummy_prob, self.from_memory)
        elif mode == 'test':
            dataset = BertSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                        self.class_num, self.rng, mode, self.dummy_prob, self.from_memory)
        else:
            raise ValueError

        return dataset


class BertSeriesDataset(AbstractDataset):
    def __init__(self,
                 dataset_path,
                 user_app_stat,
                 logtail_dict,
                 series_len,
                 class_num,
                 rng,
                 mode,
                 dummy_prob=0.35,
                 from_memory=False):
        super().__init__(dataset_path, series_len, class_num, rng, mode, from_memory)
        self.logtail_dict = logtail_dict
        self.user_app_stat = user_app_stat
        self.dummy_prob = dummy_prob

        self.dummy_class = self.class_num + 1  # 哑元,把最后一个定义为哑元，负责 mask item
        #  self.ce = nn.CrossEntropyLoss(ignore_index=ignore_item)
        self.ignore_class = self.class_num + 2  # 把倒数第二个定义为没有被mask的item的label，ce loss 会忽悠这个类的loss

    def train_data(self, hist_list):
        hist_list[hist_list >= self.class_num] = 0

        app = np.full([self.series_len], 0, dtype=np.int32)
        label = np.full([self.series_len], 0, dtype=np.int32)
        padding_mask = np.full([self.series_len], 1, dtype=np.bool_)

        idx = self.series_len - 1

        for i in reversed(hist_list):
            p = self.rng.rand()  # 随机 [0，1) 一个数
            # 有 self.dummy_prob 概率大小 item 会被 dummy
            if p < self.dummy_prob:
                p /= self.dummy_prob

                if p < 0.8:  # 用哑元替换
                    app[idx] = self.dummy_class
                elif p < 0.9:  # 用随机item替换
                    app[idx] = random_neg(i, self.class_num, self.rng)
                else:  # 保持原来的item
                    app[idx] = i

                label[idx] = i
            # 有 1 - self.dummy_prob 概率大小 item 不变，这个时候label替换为特定类别，计算损失的时候忽略这个类的loss
            else:
                app[idx] = i
                label[idx] = self.ignore_class
            padding_mask[idx] = False
            idx -= 1
            if idx == -1:
                break
        # 将最后一个app全部dummy掉
        app[-1] = self.dummy_class
        label[-1] = hist_list[-1]

        self.one_example.update({
            'app': torch.tensor(app, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long),
        })

    def val_data(self, hist_list):
        pass

    def test_data(self, hist_list):
        pass

    def add_extra_datas(self, data):
        pass


class MlpSeriesDataLoader(SasSeriesDataLoader):
    """
    dataloader 返回 app 序列信息 ,mlp 构造方式

    Args:
        SeriesDataLoader ([type]): [description]
    """
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return 'mlp_series'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = MlpSeriesDataset(self.trainset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'val':
            dataset = MlpSeriesDataset(self.valset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        elif mode == 'test':
            dataset = MlpSeriesDataset(self.testset_path, self.user_app_stat[mode], self.logtail_dict, self.series_len,
                                       self.class_num, self.rng, mode, self.from_memory)
        else:
            raise ValueError

        return dataset


class MlpSeriesDataset(AbstractDataset):
    def __init__(self, dataset_path, user_app_stat, logtail_dict, series_len, class_num, rng, mode, from_memory=False):
        super().__init__(dataset_path, series_len, class_num, rng, mode, from_memory)
        self.logtail_dict = logtail_dict
        self.user_app_stat = user_app_stat
        self.dummy_class = self.class_num + 1  # 哑元, 把最后一个定义为哑元，负责 mask item

    def train_data(self, hist_list):
        pass

    def val_data(self, hist_list):
        pass

    def test_data(self, hist_list):
        pass

    def add_extra_datas(self, data):
        pass
