import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from abc import *
from pathlib import Path

from utils import *
from loggers import *
from config import *


class AbstractTrainer(metaclass=ABCMeta):
    """
    训练器的抽象类，封装了各个模型训练相关的通用操作.

    Args:
        args ([type]): 全局参数对象
        model (object): 模型的实例化对象
        train_loader (object): 训练集加载器
        val_loader (object): 验证集加载器
        test_loader (object): 测试集加载器
        export_root (str): Log、model的存放路径
    """
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):

        self.args = args

        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1 and self.device == 'cuda'
        # 是否并行运算
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()

        # 是否动态调整学习率，使用 optim.lr_scheduler.py 包
        if args.enable_lr_schedule:
            # StepLR() 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。间隔单位是step。step通常是指epoch
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        # 计算指标，example：Top 1，2，3，4，...
        self.metric_ks = args.metric_ks
        # 基于哪个指标保存模型，example：Top 1?、2?、3?...
        self.best_metric = args.best_metric

        # 模型参数和tensorboard的保存
        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        # 默认上述记录器，如果还有新的记录器，通过 add_extra_loggers() 方法添加
        self.add_extra_loggers()
        # 将所有的记录器进行注册
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)

        # 指标数据记录的间隔
        self.log_period_as_iter = args.log_period_as_iter

        # 各类指标的集合类，返回一个字典
        self.train_average_meter_set = AverageMeterSet()
        self.val_average_meter_set = AverageMeterSet()
        self.test_average_meter_set = AverageMeterSet()

        self.epoch_start = 0
        self.accum_iter_start = 0

        self.resume_training = args.resume_training
        if self.resume_training:
            print('[*] 恢复之前的训练状态，正在加载最近的一次模型参数 ...')
            self.setup_to_resume_from_recent()
            print('[*] 完成恢复')

    def train(self):
        """
        定义了多轮训练的流程
        """
        epoch = self.epoch_start
        # 全局累加计数器，记录已经遍历过的样本数
        accum_iter = self.accum_iter_start

        for epoch in range(self.epoch_start, self.num_epochs):
            fix_random_seed_as(epoch)

            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter, mode='val')

        self.validate(epoch, accum_iter, mode='test')

        # 训练结束之后，Log 的善后
        self.logger_service.complete({
            'state_dict': (self._get_state_dict(epoch, accum_iter)),
        })

        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        """
        定义了单轮训练的流程

        Args:
            epoch (int): epoch的轮数
            accum_iter (int): 当前的总batch数，在训练程序全生命周期进行累加
        """
        self.model.train()

        # 是否动态调整学习率
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        # # 各类指标的集合类，重置
        self.train_average_meter_set.reset()

        tqdm_dataloader = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size, batch = self._batch_to_device(batch)

            self.optimizer.zero_grad()

            # 将一个batch的数据传入进行前馈运算中，返回loss和网络输出值
            scores = self.feed_forward(batch)
            loss = self.calculate_loss(batch, scores)
            metrics = self.calculate_metrics(batch, scores)

            loss.backward()
            self.optimizer.step()

            self.train_average_meter_set.update('loss', loss.item())
            for k, v in metrics.items():
                self.train_average_meter_set.update(k, v)

            accum_iter += batch_size

            # 是否需要打印训练信息到tensorboard、保存model、
            if self._needs_to_log(accum_iter):

                description_metrics = ['Acc@%d' % k for k in self.metric_ks[:]]
                description = 'Train: ' + \
                              'Epoch {}, loss {:.4f}, '.format(epoch + 1, self.train_average_meter_set['loss'].avg) + \
                              ', '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('Acc', 'A')
                description = description.format(*(self.train_average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

                log_data = {
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(self.train_average_meter_set.averages())
                self.log_extra_train_info(log_data)

                self.logger_service.log_train(log_data)
        return accum_iter

    def validate(self, epoch, accum_iter, mode='val'):
        """[summary]

        Args:
            epoch (int): epoch的轮数
            accum_iter (int): 当前的总样本数，代表 x 轴坐标点
            mode (str, optional): val or test. Defaults to 'val'.
        """

        if mode == 'val':
            loader = self.val_loader
            average_meter_set = self.val_average_meter_set
        elif mode == 'test':
            loader = self.test_loader
            average_meter_set = self.test_average_meter_set
            print('用测试集测试最佳模型!')
            best_model = torch.load(os.path.join(self.export_root, 'models',
                                                 BEST_STATE_DICT_FILENAME)).get(STATE_DICT_KEY)
            if self.is_parallel:
                self.model.module.load_state_dict(best_model)
            else:
                self.model.load_state_dict(best_model)
        else:
            raise ValueError

        self.model.eval()

        average_meter_set.reset()

        with torch.no_grad():
            tqdm_dataloader = tqdm(loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                _, batch = self._batch_to_device(batch)
                # 将一个batch的数据传入指标计算函数中，返回指标值
                scores = self.feed_forward(batch)
                metrics = self.calculate_metrics(batch, scores)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                description_metrics = ['Acc@%d' % k for k in self.metric_ks[:]]
                description = 'Val: ' if mode == 'val' else 'Test: '
                description = description + ', '.join(s + ' {:.5f}' for s in description_metrics)
                description = description.replace('Acc', 'A')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

        average_metrics = average_meter_set.averages()

        if mode == 'val':
            log_data = {
                'state_dict': (self._get_state_dict(epoch + 1, accum_iter)),
                'epoch': epoch + 1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_metrics)
            self.log_extra_val_info(log_data)
            # 验证集跑完之后记录最后的平均准确率
            self.logger_service.log_val(log_data)
        elif mode == 'test':
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)
        else:
            raise ValueError

    def _batch_to_device(self, batch):
        """
        在此处将一个 batch 的数据 转换为模型需要的格式的数据，并加载到device中
        """
        batch_size = list(batch.values())[0].size(0)
        batch = {k: v.to(self.device) for (k, v) in batch.items()}
        return batch_size, batch

    def _create_optimizer(self):
        """
        定义优化器
        """
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        """
        创建记录器，负责tensorboard writer 和 模型 的写入工作
        """
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricTensorBoardPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricTensorBoardPrinter(writer, key='loss', graph_name='Total_Loss', group_name='Train'),
        ]
        for k in self.metric_ks:
            train_loggers.append(
                MetricTensorBoardPrinter(writer, key='Acc@%d' % k, graph_name='Acc@%d' % k, group_name='Train'))

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricTensorBoardPrinter(writer, key='Acc@%d' % k, graph_name='Acc@%d' % k, group_name='Validation'))

        val_loggers.append(RecentModelLogger(model_checkpoint, filename=RECENT_STATE_DICT_FILENAME))
        val_loggers.append(
            BestModelLogger(model_checkpoint, metric_key=self.best_metric, filename=BEST_STATE_DICT_FILENAME))
        return writer, train_loggers, val_loggers

    def _get_state_dict(self, epoch, accum_iter):
        """
        默认同时保存模型和优化器的参数，通过dict保存

        Returns:
            dict: [description]
        """
        dict__ = {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }
        if self.args.enable_lr_schedule:
            dict__.update({SCHEDULER_STATE_DICT_KEY: self.lr_scheduler.state_dict()})

        return dict__

    def setup_to_resume_from_recent(self):
        """
        从最近的保存点继续训练

        Args:
            args ([type]): [description]
            model ([type]): [description]
            optimizer ([type]): [description]
        """

        chk_dict = torch.load(os.path.join(self.export_root, 'models', RECENT_STATE_DICT_FILENAME))

        if self.is_parallel:
            self.model.module.load_state_dict(chk_dict[STATE_DICT_KEY])
        else:
            self.model.load_state_dict(chk_dict[STATE_DICT_KEY])

        if OPTIMIZER_STATE_DICT_KEY in chk_dict:
            self.optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])
        if self.args.enable_lr_schedule and SCHEDULER_STATE_DICT_KEY in chk_dict:
            self.lr_scheduler.load_state_dict(chk_dict[SCHEDULER_STATE_DICT_KEY])
        self.epoch_start, self.accum_iter_start = chk_dict[STEPS_DICT_KEY]

    def _get_ignore_class(self):
        """
        计算 类别排序的 mask，值为 0 的位置的class不参与 tok k 排序
            example：ignore_class=[1,3]--->ignore_idxs = [1, 0, 1, 0, 1, 1, 1, 1]，类别 1 和 3 不参与 top k 计算

        Returns:
            [type]: [description]
        """
        ingore_idxs = torch.ones(self.args.class_num).to(self.device)
        if self.args.ignore_class is not None and len(self.args.ignore_class) != 0:
            for i in self.args.ignore_class:
                ingore_idxs[i] = 0
        return ingore_idxs

    @abstractmethod
    def add_extra_loggers(self):
        """
        添加额外的log记录器，默认的记录器 在 _create_loggers() 方法中
        """
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def feed_forward(self, batch):
        """
        将一个batch的数据传入进行前馈运算中，返回 logits 和 预测值

        Args:
            batch (dataset): 传入一个 batch 的数据
        """
        return self.model(batch)

    @abstractmethod
    def calculate_loss(self, batch, scores):
        """
        计算 Loss

        Args:
            batch (dataset): 传入一个 batch 的数据，和 feed_forward 输出值
        """

    @abstractmethod
    def calculate_metrics(self, batch, scores):
        """
        计算评价指标

        Args:
            batch (dataset): 传入一个 batch 的数据，和 feed_forward 输出值
        """
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        """
        更新train过程中的默认写入log的信息

        Args:
            log_data (dict): log信息
        """
        pass

    @abstractmethod
    def log_extra_val_info(self, log_data):
        """
        更新val过程中的默认写入log的信息

        Args:
            log_data (dict): log信息
        """
        pass

    def _needs_to_log(self, accum_iter):
        """
        是否需要输出训练数据到 tensorboard 的判定规则，每隔 log_period_as_iter 次，打印一次
        数据集必须使用drop_last = true
        https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        Args:
            accum_iter (int): 样本数:  n*batch_size, n 为 dataloader enumerate 遍历次数

        Returns:
            [bool]: [description]
        """
        return accum_iter != 0 and (accum_iter % (self.log_period_as_iter * self.args.train_batch_size)) == 0
