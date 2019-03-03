import os
import json

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def define_common_flags():
    """Define common flags."""
    # common flags
    flags.DEFINE_integer('batch_size', 1, 'Batch size.')
    flags.DEFINE_integer('crop_width', None, 'Width of the central crop for images.')
    flags.DEFINE_integer('crop_height', None, 'Height of the central crop for images.')
    flags.DEFINE_string('train_log_dir', 'my_logs',  # default: logs
                        'Directory where to write event logs.')
    flags.DEFINE_string('dataset_name', 'van', 'Name of the dataset. Supported: fsns')
    flags.DEFINE_string('model_name', 'model', 'Name of the model.')
    flags.DEFINE_string('split_name', 'train', 'Dataset split name to run evaluation for: test,train.')
    flags.DEFINE_string('data_root', None, 'Data root folder.')
    flags.DEFINE_string('checkpoint', '', 'Path for checkpoint to restore weights from.')
    flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
    flags.DEFINE_bool('do_augment', False, '')

    # Model hyper parameters
    flags.DEFINE_float('learning_rate', 0.004, 'learning rate')
    flags.DEFINE_string('optimizer', 'momentum', 'the optimizer to use')
    flags.DEFINE_float('momentum', 0.9, 'momentum value for the momentum optimizer if used')
    flags.DEFINE_bool('use_augment_input', True, 'If True will use image augmentation')

    # Method hyper parameters
    # conv_tower_fn
    flags.DEFINE_string('final_endpoint', 'Mixed_5d', 'Endpoint to cut inception tower')

    # sequence_logit_fn
    flags.DEFINE_bool('use_attention', True, 'If True will use the attention mechanism')
    flags.DEFINE_bool('use_autoregression', True, 'If True will use autoregression (a feedback link)')
    flags.DEFINE_integer('num_lstm_units', 256, 'number of LSTM units for sequence LSTM')
    flags.DEFINE_float('weight_decay', 0.00004, 'weight decay for char prediction FC layers')
    flags.DEFINE_float('lstm_state_clip_value', 10.0,
                       'cell state is clipped by this value prior to the cell output activation')

    # 'sequence_loss_fn'
    flags.DEFINE_float('label_smoothing', 0.1, 'weight for label smoothing')
    flags.DEFINE_bool('ignore_nulls', True, 'ignore null characters for computing the loss')
    flags.DEFINE_bool('average_across_timesteps', False, 'divide the returned cost by the total label weight')
    flags.DEFINE_bool('use_location', False, 'If true will use location attention')


def print_flags(filename=None):
    info = '-' * 35 + 'FLAGS' + '-' * 35 + '\n'
    flag_dict = {}

    ignore_names = {'help', 'helpshort', 'helpfull'}  # Set literal construction

    for _, flag in sorted(FLAGS._flags().items()):

        if flag.name in ignore_names:
            continue

        comment = ''
        if flag.value != flag.default:
            comment = '\t[default: {}]'.format(flag.default)
        info += '{:>20}:  {:<30}{}\n'.format(flag.name, str(flag.value), comment)

        flag_dict[flag.name] = str(flag.value)
    info += '-' * 75 + '\n'

    print(info)

    if filename:
        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(flag_dict, file, indent=4)
