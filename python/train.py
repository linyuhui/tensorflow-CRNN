import logging
import collections
import shutil
import tensorflow as tf
from tensorflow.contrib import slim

from utils import data_loader
import os
import datasets
import models

from utils.tf_utils import define_common_flags, print_flags

flags = tf.flags
FLAGS = flags.FLAGS

define_common_flags()
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training with '
                                'multiple workers to identify each worker.')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers. If the value is 0, then'
                                    ' the parameters are handled locally by the worker.')
flags.DEFINE_integer('save_summaries_secs', 60, 'The frequency with which summaries are saved, in '
                                                'seconds.')
flags.DEFINE_integer('save_interval_secs', 600, 'Frequency in seconds of saving the model.')

flags.DEFINE_float('clip_gradient_norm', 2.0, 'If greater than 0 then the gradients would be clipped by '
                                              'it.')
flags.DEFINE_bool('sync_replicas', False, 'If True will synchronize replicas during training.')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of gradients updates before updating params.')
flags.DEFINE_integer('total_num_replicas', 1, 'Total number of worker replicas.')
flags.DEFINE_integer('startup_delay_steps', 15, 'Number of training steps between replicas startup.')
flags.DEFINE_boolean('reset_train_dir', False, 'If true will delete all files in the train_log_dir')
flags.DEFINE_boolean('show_graph_stats', False, 'Output model size stats to stderr.')
flags.DEFINE_float('pre_gpu_mem', 0.55, 'per_process_gpu_memory_fraction')
flags.DEFINE_string('gpus', '', '')

flags.DEFINE_integer('log_every_n_steps', 1, '')
flags.DEFINE_integer('max_number_of_steps', int(1e5),
                     'The maximum number of gradient steps.')


def get_training_hparams():
    TrainingHParams = collections.namedtuple('TrainingHParams', ['learning_rate', 'optimizer', 'momentum', 'do_augment'])

    return TrainingHParams(
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum,
        do_augment=FLAGS.do_augment)


def create_optimizer(hparams):
    """Creates optimized based on the specified flags.
    Args:
        hparams: A TrainingHParams."""
    optimizer = None
    if hparams.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    return optimizer


def train(loss, init_fn, hparams):
    """Wraps slim.learning.train to run a training loop.

    Args:
      loss: a loss tensor
      init_fn: A callable to be executed after all other initialization is done.
      hparams: a model hyper parameters
    """
    optimizer = create_optimizer(hparams)

    if FLAGS.sync_replicas:
        # replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
        optimizer = tf.train.SyncReplicasOptimizer(opt=optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                                                   total_num_replicas=FLAGS.total_num_replicas)

        sync_optimizer = optimizer
        startup_delay_steps = 0
    else:
        startup_delay_steps = 0
        sync_optimizer = None
    # set GPU option 
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = FLAGS.pre_gpu_mem

    # If checkpoint has save all variable, this function will get global step from
    # tf.GraphDef.GLOBAL_STEP or `graph.get_tensor_by_name('global_step:0')`
    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=True,
        clip_gradient_norm=FLAGS.clip_gradient_norm)

    saver = tf.train.Saver(max_to_keep=10)

    # This function include init_op and local_init_op, init_op include global_variables_initializer.
    # local_init_op include local and table initializer.
    slim.learning.train(
        train_op=train_op,
        logdir=FLAGS.train_log_dir,
        graph=loss.graph,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        startup_delay_steps=startup_delay_steps,
        sync_optimizer=sync_optimizer,
        init_fn=init_fn,
        session_config=config,
        saver=saver,
        log_every_n_steps=FLAGS.log_every_n_steps
    )


def prepare_training_dir():
    if not os.path.exists(FLAGS.train_log_dir):
        logging.info('Create a new training directory %s', FLAGS.train_log_dir)
        os.makedirs(FLAGS.train_log_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info('Reset the training directory %s', FLAGS.train_log_dir)
            shutil.rmtree(FLAGS.train_log_dir)
            os.makedirs(FLAGS.train_log_dir)
        else:
            logging.info('Use already existing training directory %s',
                         FLAGS.train_log_dir)


def profile_graph():
    stats = tf.profiler.profile(tf.get_default_graph())
    return stats


def main(_):
    print_flags()
    import logging
    import sys
    from tensorflow.python.platform import tf_logging

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stderr,
        format='%(levelname)s '
               '%(asctime)s.%(msecs)06d: '
               '%(filename)s: '
               '%(lineno)d '
               '%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    tf_logger = tf_logging._get_logger()
    tf_logger.propagate = False

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    prepare_training_dir()
    dataset = datasets.create_dataset()

    model = models.create_model(num_char_classes=dataset.num_char_classes,  # represents `num_labels + 1` classes
                                max_seq_len=dataset.max_seq_len,
                                null_code=dataset.null_code)
    hparams = get_training_hparams()

    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.

    device_setter = tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)
    with tf.device(device_setter):
        data = data_loader.get_data(dataset)

        endpoints = model.create_base(data.images, is_training=True)

        total_loss = model.create_loss(data, endpoints)
        init_fn = model.create_init_fn(FLAGS.checkpoint)

        # print(tf.trainable_variables('CRNN'))
        if FLAGS.show_graph_stats:
            logging.info('Total number of weights in the graph: %s',
                         profile_graph())
        train(total_loss, init_fn, hparams)


if __name__ == '__main__':
    tf.app.run()
