from .abstract import AbstractDataloader
from torchvision import datasets, transforms


class MnistDataLoader(AbstractDataloader):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def code(cls):
        return 'mnist'

    def _get_dataset(self, mode):
        if mode == 'train':
            dataset = datasets.MNIST(root='./0_code_trash/mnist_dataset/',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
        elif mode == 'val':
            dataset = datasets.MNIST(root='./0_code_trash/mnist_dataset/',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)
        elif mode == 'test':
            dataset = datasets.MNIST(root='./0_code_trash/mnist_dataset/',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)
        else:
            raise ValueError

        return dataset
