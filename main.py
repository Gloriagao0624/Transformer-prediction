import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from analyzers import analyzer_factory
from utils import *


def train():
    export_root = setup_run(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


def analyzer():
    import_root = setup_run(args)
    loader = dataloader_factory(args)
    model = model_factory(args)
    analyzer = analyzer_factory(args, model, loader, import_root)
    analyzer.analyse()


if __name__ == '__main__':
    if args.run_mode == 'train':
        train()
    elif args.run_mode == 'analyse':
        analyzer()
    else:
        raise ValueError('[!] "args.mode" 运行模式错误， choise=[\'train\', \'analyse\']！')
