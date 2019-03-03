import os
import re
import tensorflow as tf
import logging
import collections

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

Dataset = collections.namedtuple('Dataset', [
    'dataset', 'num_char_classes', 'max_seq_len', 'null_code'
])

DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(__file__), '')

# The dataset configuration, should be used only as a default value.
DEFAULT_CONFIG = {
    'name': 'text',
    'splits': {
        'train': {
            'size': 100,
            'pattern': 'tf_record/train*'
        },
        'test': {
            'size': 1,
            'pattern': 'tf_record/test*'
        },
        'validation': {
            'size': 1,
            'pattern': 'tf_record/validation*'
        }
    },
    'vocab_filename': 'vocab69.txt',
    'image_shape': (10, 10, 3),
    'max_sequence_length': 8,
    'null_code': 27,  # code of null char (max code)
    'items_to_descriptions': {
        'image': 'A [H x W x C] color image.',
        'label': 'Characters codes.',
        'text': 'A unicode string.',
        'length': 'A length of the encoded text.',
    }
}


def read_code_vocab(filename, null_character=u'\u2591'):
    pattern = re.compile(r'(\d+)\t(.+)')
    code_vocab = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                # Skips blank char, otherwise code is 0 and one-hot vector is same as first zeros input.
                if i == 0:
                    continue
                logging.warning('incorrect vocab file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            code_vocab[code] = char
    return code_vocab


def get_crop_size(args):
    if 'crop_width' in vars(args).keys() and 'crop_height' in vars(args).keys():
        return args.crop_width, args.crop_height
    else:
        return None


def usual_preprocess_image(image):
    with tf.variable_scope('PreprocessImage'):
        assert image.dtype == tf.float32
        image = tf.divide(image, 255.0)
        image = tf.subtract(image, 0.5)

    return image


def get_split(config=None):
    split_name = FLAGS.split_name
    batch_size = FLAGS.batch_size

    if FLAGS.data_root:
        data_root = FLAGS.data_root
    else:
        data_root = DEFAULT_DATA_ROOT

    if not config:
        config = DEFAULT_CONFIG

    if split_name not in config['splits']:
        raise ValueError('Split name %s was not recognized.' % split_name)

    logging.info('Using %s dataset split_name=%s data_root=%s', config['name'],
                 split_name, data_root)

    def _parse_fn(example_proto):
        # Ignores the 'image/height' feature.
        zero = tf.zeros([1], dtype=tf.int64)
        # Choose the corresponding way to parse feature.
        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature([], tf.string, default_value='png'),
            'image/width':
                tf.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/orig_width':
                tf.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/class':
                tf.FixedLenFeature([config['max_sequence_length']], tf.int64),
            'image/unpadded_class':
                tf.VarLenFeature(tf.int64),
            'image/text':
                tf.FixedLenFeature([1], tf.string, default_value=''),
            'image/text_len': tf.FixedLenFeature([], tf.int64)  # sequence_length in ctc_loss must be rank 1
        }
        features = tf.parse_single_example(example_proto, features=keys_to_features)
        # bs = features['image/encoded'].shape[0]
        # print('bs' , bs)
        image = tf.decode_raw(features['image/encoded'], out_type=tf.uint8)
        shape = config['image_shape']
        image = tf.cast(image, dtype=tf.float32)
        image = tf.reshape(image, shape)
        label = features['image/unpadded_class']
        label = tf.cast(label, tf.int32)
        seq_len = features['image/text_len']
        seq_len = tf.cast(seq_len, tf.int32)
        image = usual_preprocess_image(image)

        return image, label, seq_len

    # Preprocess 4 files concurrently.
    file_pattern = os.path.join(data_root, config['splits'][split_name]['pattern'])
    files = tf.data.Dataset.list_files(file_pattern)
    # dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda fname: tf.data.TFRecordDataset(fname), cycle_length=4
        )
    )
    dataset = dataset.map(_parse_fn, num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    vocab_file = os.path.join(os.path.dirname(__file__), config['vocab_filename'])
    code_vocab = read_code_vocab(vocab_file)
    return Dataset(dataset=dataset,
                   num_char_classes=max(code_vocab.keys()) + 1,  # represents `num_labels + 1` classes (CTC)
                   max_seq_len=config['max_sequence_length'],
                   null_code=config['null_code'])  # num_char_classes add 1 for ctc_loss
