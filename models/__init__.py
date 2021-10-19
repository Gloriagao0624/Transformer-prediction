from .abstract import AbstractModel
from utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractModel)

MODELS = {c.code(): c for c in all_subclasses(AbstractModel) if c.code() is not None}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
