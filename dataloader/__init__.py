from .abstract import AbstractDataloader
from utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractDataloader)

DATALOADERS = {c.code(): c for c in all_subclasses(AbstractDataloader) if c.code() is not None}


def dataloader_factory(args):
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args)

    if args.run_mode == 'train':
        train, val, test = dataloader.get_dataloaders()
        return train, val, test
    elif args.run_mode == 'analyse':
        test = dataloader.get_analyse_dataloader()
        return test
    else:
        raise ValueError('[!] "args.mode" 训练模式错误！')
