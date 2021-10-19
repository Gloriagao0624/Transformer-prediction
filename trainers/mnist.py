import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from .abstract import *
from .utils.metrics import accuracy_topk


class MnistTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.task_loss = F.nll_loss

    @classmethod
    def code(cls):
        return 'mnist'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def _batch_to_device(self, batch):
        batch_size = batch[0].size(0)
        batch = [batch[0].to(self.device), batch[1].to(self.device)]  # mnist
        return batch_size, batch

    def calculate_loss(self, batch, scores):
        _, label = batch
        output_logits = scores['y_pred']
        # time.sleep(5)
        loss = self.task_loss(output_logits, label)
        return loss

    def calculate_metrics(self, batch, scores):
        _, label = batch
        y_pred = scores['y_pred']
        metrics = accuracy_topk(y_pred, label, ks=self.metric_ks,name='Acc')['metrics']
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
