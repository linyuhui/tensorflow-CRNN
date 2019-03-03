import numpy as np
import PIL.Image

import tensorflow as tf

import datasets
import time
import os
import models
import importlib

from utils.tf_utils import print_flags, define_common_flags
from utils.misc import read_code_vocab

# no gpu
os.environ['CUDA_VISIBLE_DEVICES'] = ''

define_common_flags()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', 'sample_images', '')
flags.DEFINE_string('output_node_names', '', 'Names of output nodes (operation).')
flags.DEFINE_bool('freeze_graph', False, '')
flags.DEFINE_string('vocab_path', '/home/yhlin/projects/tensorflow-CRNN/python/datasets/vocab69.txt', '')
flags.DEFINE_integer('max_seq_len', 12, '')
flags.DEFINE_string('result_dir', '', '')


def get_dataset_image_size():
    # Ideally this info should be exposed through the dataset interface itself.
    # But currently it is not available by other means.
    module_name = 'datasets.' + FLAGS.dataset_name
    ds_module = importlib.import_module(module_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']

    return width, height


def get_image_labels(image_dir, check=False):
    import os
    count = 0
    filenames = []
    labels = []
    for f in os.listdir(image_dir):
        try:
            if not f.endswith(('.gif', '.jpg', '.png', 'bmp')):
                continue
            fp = os.path.join(image_dir, f)
            if not os.path.isabs(fp):
                fp = os.path.abspath(fp)
            if not os.path.exists(fp):
                continue
            if check:
                PIL.Image.open(fp)
                # cv2.imread(fp)
            image_name = f.split('_')[1]
            filenames.append(fp)
            labels.append(image_name)
            count += 1
        except Exception as e:
            print("fn:%s,error: %s", fp, e)
            os.remove(fp)
    return filenames, labels


def get_images(image_dir, check=False):
    import os
    count = 0
    fnames = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:

            try:
                if not file.endswith(('.png', 'bmp', '.jpg')):
                    continue
                fp = os.path.join(root, file)
                if not os.path.isabs(fp):
                    fp = os.path.abspath(fp)
                    if not os.path.exists(fp):
                        continue
                if check:
                    PIL.Image.open(fp)
                fnames.append(fp)
                count += 1
            except Exception as e:
                print("fn:%s,error: %s", file, e)
    print('Get {} images'.format(count))

    return fnames


def create_model():
    width, height = get_dataset_image_size()
    dataset = datasets.create_dataset()

    model = models.create_model(num_char_classes=dataset.num_char_classes,
                                max_seq_len=dataset.max_seq_len,
                                null_code=dataset.null_code)

    images_placeholder = tf.placeholder(tf.float32, shape=[1, height, width, 3])
    images = images_placeholder
    endpoints = model.create_base(images, is_training=False)
    return images_placeholder, endpoints


def preprocess_image(image):
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - 0.5
    return image


def load_image(filename):
    width, height = get_dataset_image_size()
    print('Image size: {}x{}'.format(height, width))
    image = PIL.Image.open(filename).resize([width, height], PIL.Image.BILINEAR)
    image_data = np.ndarray([1, height, width, 3], dtype=np.uint8)
    image_data[0, ...] = np.asarray(image)
    return image_data


def parse_prediction(prediction):
    vocab = read_code_vocab(FLAGS.vocab_path)
    result = ''
    for i in range(prediction.shape[1]):
        result += vocab[prediction[0, i]]

    return result


def run(checkpoint, image_dir):
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    images_placeholder, endpoints = create_model()

    if tf.__version__ == '1.12.0':
        dense_decoded = tf.sparse.to_dense(endpoints.decoded[0], default_value=68)
    else:
        dense_decoded = tf.sparse_to_dense(
            sparse_values=endpoints.decoded[0].values,
            sparse_indices=endpoints.decoded[0].indices,
            default_value=68,
            output_shape=[FLAGS.batch_size, FLAGS.max_seq_len],
        )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        for fp in get_images(image_dir):
            image_data = load_image(fp)
            image_data = preprocess_image(image_data)

            start_time = time.time()
            prediction = sess.run([dense_decoded], feed_dict={images_placeholder: image_data})
            print('{} seconds'.format(time.time() - start_time))

            image_name = os.path.basename(fp)
            import shutil
            shutil.copy(fp, os.path.join(FLAGS.result_dir, image_name))

            name, _ = os.path.splitext(image_name)

            text_predicted = parse_prediction(prediction[0])
            with open(os.path.join(FLAGS.result_dir, name + '.txt'), 'w', encoding='utf-8') as file:
                file.write(text_predicted)


def main(_):
    print_flags()
    run(FLAGS.checkpoint, FLAGS.image_dir)


if __name__ == '__main__':
    tf.app.run()
