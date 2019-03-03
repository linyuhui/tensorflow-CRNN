import importlib
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def create_dataset():
    module_name = 'datasets.' + FLAGS.dataset_name
    ds_module = importlib.import_module(module_name)

    return ds_module.get_split()

