import torch.nn as nn

from abc import *


class AbstractModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass