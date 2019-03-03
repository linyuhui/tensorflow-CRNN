import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
import models
import datasets

from utils import data_loader
from utils.tf_utils import define_common_flags, print_flags

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

FLAGS = flags.FLAGS
define_common_flags()

flags.DEFINE_integer('num_batches', 1,
                     'Number of batches to run eval for.')
flags.DEFINE_string('eval_log_dir', 'eval_logs',
                    'Directory where the evaluation results are saved to.')
flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency in seconds to run evaluations.')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of times to run evaluation.')
# Eval once.
flags.DEFINE_string('ckpt_path', '', '')
flags.DEFINE_string('eval_type', 'once', 'Evaluete once or loop.')


def main(_):
    import logging
    import sys
    from tensorflow.python.platform import tf_logging

    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stderr,
                        format='%(levelname)s '
                                                                       '%(asctime)s.%(msecs)06d: '
                                                                       '%(filename)s: '
                                                                       '%(lineno)d '
                                                                       '%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    tf_logger = tf_logging._get_logger()
    tf_logger.propagate = False

    print_flags()

    if not tf.gfile.Exists(FLAGS.eval_log_dir):
        tf.gfile.MakeDirs(FLAGS.eval_log_dir)

    dataset = datasets.create_dataset()
    model = models.create_model(num_char_classes=dataset.num_char_classes,
                                max_seq_len=dataset.max_seq_len,
                                null_code=dataset.null_code)

    data = data_loader.get_data(dataset)
    endpoints = model.create_base(data.images, is_training=False)
    eval_ops, prediction, label = model.create_eval_ops(
        data, endpoints)

    tf.train.get_or_create_global_step()

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    if FLAGS.eval_type == 'once':
        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=FLAGS.ckpt_path,
            logdir=FLAGS.eval_log_dir,
            num_evals=FLAGS.num_batches,
            eval_op=eval_ops,
            session_config=session_config
        )
    elif FLAGS.eval_type == 'loop':
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.train_log_dir,
            logdir=FLAGS.eval_log_dir,
            eval_op=eval_ops,
            num_evals=FLAGS.num_batches,
            eval_interval_secs=FLAGS.eval_interval_secs,
            max_number_of_evaluations=FLAGS.number_of_steps,
            timeout=2000,
            session_config=session_config)
    else:
        pass


if __name__ == '__main__':
    app.run()
