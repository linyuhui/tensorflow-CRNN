import importlib

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def create_model(*args, **kwargs):
    module_name = 'models.' + FLAGS.model_name
    module = importlib.import_module(module_name)

    model = None
    for name, cls in module.__dict__.items():
        if name.lower() == FLAGS.model_name.lower():
            model = cls
    model = model(*args, **kwargs)
    return model
