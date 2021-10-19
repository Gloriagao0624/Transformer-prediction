import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstract import *
from .utils.metrics import accuracy_topk_add_bias
from .utils.loss import SmoothTopkSVM


class E2ETrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

        self.task_loss = SmoothTopkSVM(n_classes=args.class_num, alpha=1., tau=1., k=max(args.metric_ks))
        self.task_loss.cpu() if self.device == 'cpu' else self.task_loss.cuda()

        # self.task_loss = F.cross_entropy
        self.wechat_ratio = args.dataset_wechat_ratio
        self.ingore_classes = self._get_ignore_class()

    @classmethod
    def code(cls):
        return 'e2e'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch, scores):
        label = batch['label']
        output_logits = scores['output_logits']
        loss = self.task_loss(output_logits, label)
        return loss

    def calculate_metrics(self, batch, scores):
        label = batch['label']
        y_pred = scores['y_pred']
        metrics = accuracy_topk_add_bias(y_pred,
                                         label,
                                         name='Acc',
                                         ks=self.metric_ks,
                                         bias=self.wechat_ratio,
                                         ignore_idxs=self.ingore_classes)['metrics']

        return metrics

    def _get_state_dict(self, epoch, accum_iter):
        """
        获取模型的参数，通过dict保存

        Returns:
            dict: [description]
        """
        return {
            # 保存模型model的参数
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }
