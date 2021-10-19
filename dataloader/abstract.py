from abc import *
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

import linecache
import os


class AbstractDataloader(metaclass=ABCMeta):
    """Dataloader抽象类

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """
    def __init__(self, args):
        self.args = args
        self.class_num = args.class_num
        self.series_len = args.series_len
        self.from_memory = args.from_memory

        self.trainset_path = args.trainset_path
        self.valset_path = args.valset_path
        self.testset_path = args.testset_path

        self.data_path = {'train': self.trainset_path, 'val': self.valset_path, 'test': self.testset_path}

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size

        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_dataloaders(self):
        train_loader = self._loader('train')
        val_loader = self._loader('val')
        test_loader = self._loader('test')
        return train_loader, val_loader, test_loader

    def get_analyse_dataloader(self):
        return self._loader('val')

    def _loader(self, mode):
        batch_size = {'train': self.train_batch_size, 'val': self.val_batch_size, 'test': self.test_batch_size}[mode]
        dataset = self._get_dataset(mode)
        shuffle = True

        # 如果不从内存读取数据集，则需要设置 drop_last = False，因为要让迭代器读取到文件末尾从而触发 close()
        # 否则，就会提前 drop last 导致提前退出迭代器，下一轮的时候就会出现文件不是从头开始读的问题
        drop_last = True if self.from_memory else False

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    @abstractmethod
    def _get_dataset(self, mode):
        pass


class AbstractDataset(Dataset, metaclass=ABCMeta):
    """Dataset 抽象类

    Args:
        Dataset ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """
    def __init__(self, dataset_path, series_len, class_num, rng, mode, from_memory):
        self.dataset_path = dataset_path
        self.series_len = series_len
        self.from_memory = from_memory
        self.class_num = class_num
        self.rng = rng
        self.mode = mode

        self.example_num = int(os.popen(f'wc -l {self.dataset_path}').read().split()[0])
        print('[*] 完成数据集：{}  样本数统计，总计  {}  条样本'.format(self.dataset_path, self.example_num))

        self._begin_read_file()

        self.keys = [
            'uid', 'phone_model', 'signal_types', 'app', 'app_name', 'timestamp', 'duration', 'week', 'hour', 'month',
            'delta_t'
        ]

        # 存放一条样本
        self.one_example = {}

    def _begin_read_file(self):
        if not self.from_memory:
            self.sum_counter = 0  # 计数器
            self.dataset_file = open(self.dataset_path)

    def _read_one_sample(self, index):
        self.one_example = {}
        data = {}
        if not self.from_memory:
            self.sum_counter += 1
            if self.sum_counter == self.example_num:
                self.dataset_file.close()
                raise StopIteration
            # index 实际上没有真正用上
            values = self.dataset_file.readline().strip('\n').split('\t')
        else:
            values = linecache.getline(self.dataset_path, index + 1).strip('\n').split('\t')  # 去除行尾换行符，再根据'\t'切分

        for (key, value) in zip(self.keys, values):
            value = value.split(',')
            if value == '' or value is None:
                line = index if self.from_memory else self.example_num
                print('[!] 第 {} 行出现了空值，请检查数据集！'.format(line))
                raise ValueError
            data[key] = value[0] if len(value) == 1 else value

        return data

    def _reset_read_file(self):
        if not self.from_memory and self.dataset_file.closed:
            self.sum_counter = 0  # 计数器
            self.dataset_file = open(self.dataset_path)

    def build_item_data(self, data):
        """
        app 序列数据的构造
        """
        # App使用序列
        hist_list = [int(v) for v in data['app']]
        hist_list = np.array(hist_list)

        if self.mode == 'train':
            self.train_data(hist_list)
        elif self.mode == 'val':
            self.val_data(hist_list)
        elif self.mode == 'test':
            self.test_data(hist_list)
        else:
            raise ValueError

    def __getitem__(self, index):
        raw_data = self._read_one_sample(index)
        self.build_item_data(raw_data)
        self.add_extra_datas(raw_data)
        return self.one_example

    def __len__(self):
        self._reset_read_file()
        return self.example_num

    @abstractmethod
    def add_extra_datas(self, data):
        """
        添加额外数据，比如时间数据，比如用户embed数据
        """
        pass

    @abstractmethod
    def train_data(self, hist_list):
        """
        训练集数据构造
        """
        pass

    @abstractmethod
    def val_data(self, hist_list):
        """
        验证集数据构造
        """
        pass

    @abstractmethod
    def test_data(self, hist_list):
        """
        测试集数据构造
        """
        pass