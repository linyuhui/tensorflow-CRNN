import tensorflow as tf
import collections
import functools
import logging
import sys

from utils import misc
from tensorflow.contrib import slim
from tensorflow.python.platform import flags
from models.cnn_ops import tf_conv2d, tf_max_pooling2d, batch_norm
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from utils import metrics

FLAGS = flags.FLAGS

OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'chars_logit', 'decoded'
])

DataParams = collections.namedtuple('DataParams', [
    'num_char_classes', 'max_seq_len', 'null_code'
])

ModelHParams = collections.namedtuple('ModelHParams', [
    'hidden_size', 'num_layers', 'feature_map_seq_len'
])


class CRNN:
    """
        Implement the crnn model for sequence recognition
    """

    def __init__(self, num_char_classes, max_seq_len, null_code, hparams=None):

        super(CRNN, self).__init__()
        self._dparams = DataParams(
            num_char_classes=num_char_classes,
            max_seq_len=max_seq_len,
            null_code=null_code
        )
        default_hparams = ModelHParams(
            hidden_size=256,
            num_layers=2,
            feature_map_seq_len=16
        )
        self._hparams = default_hparams
        if hparams:
            self._hparams.__dict__.update(hparams)

    def create_base(self, images, is_training, scope='CRNN', reuse=None):
        logging.debug('Input images: {}'.format(images))
        with tf.variable_scope(scope, reuse=reuse):
            net = self._conv_tower_fn(images, is_training=is_training)
            logging.debug('Conv tower: %s', net)
            net = self._map_to_sequence(net)
            logging.debug('To sequence: {}'.format(net))
            logits = self._sequence_fn(net, is_training=is_training)
            logging.debug('Sequence logits: {}'.format(logits))
            # Parameter `sequence_length` is length of rnn outputs.
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits,
                tf.cast(tf.fill([FLAGS.batch_size], self._hparams.feature_map_seq_len), dtype=tf.int32),
                merge_repeated=False
            )
            return OutputEndpoints(chars_logit=logits, decoded=decoded)

    def create_loss(self, data, endpoints):
        self._loss_fn(data.labels, endpoints.chars_logit,
                      tf.cast(tf.fill([FLAGS.batch_size], self._hparams.feature_map_seq_len), dtype=tf.int32))
        # self._loss_fn(data.labels, endpoints.chars_logit, data.sequence_length)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    @staticmethod
    def _loss_fn(labels, logits, seq_len):
        # Parameter `sequence_lengths` of ctc_loss is the length of rnn outputs.
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))
        tf.losses.add_loss(loss)

    def _conv_tower_fn(self, inputs, is_training, reuse=None):
        """Computes convolutional feautures.

        Args:
            inputs: A tensor of shape [batch_size, 32, 192, 3]
        """
        with tf.variable_scope('conv_tower_fn'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            conv2d_fn = functools.partial(tf_conv2d, kernel_size=3, stride=1, use_bias=False)
            net = conv2d_fn(inputs, out_channels=64, name='conv1')
            net = tf.nn.relu(net)
            net = tf_max_pooling2d(net, kernel_size=[2, 1], stride=[2, 1])
            net = conv2d_fn(net, out_channels=128, name='conv2')
            net = tf.nn.relu(net)
            net = tf_max_pooling2d(net, kernel_size=2, stride=2)
            net = conv2d_fn(net, out_channels=256, name='conv3')
            net = tf.nn.relu(net)
            net = conv2d_fn(net, out_channels=256, name='conv4')
            net = tf.nn.relu(net)
            net = tf_max_pooling2d(net, kernel_size=[2, 2], stride=[2, 2])
            net = conv2d_fn(net, out_channels=512, name='conv5')
            net = tf.nn.relu(net)
            net = batch_norm(net, is_training=is_training)
            net = conv2d_fn(net, out_channels=512, name='conv6')
            net = tf.nn.relu(net)
            net = batch_norm(net, is_training=is_training)
            net = tf_max_pooling2d(net, kernel_size=[2, 1], stride=[2, 1])
            net = self.conv2d(net, out_channels=512, kernel_size=2, stride=[2, 2], use_bias=False, name='conv7')
            net = tf.nn.relu(net)
            return net

    @staticmethod
    def _map_to_sequence(inputs):
        assert inputs.get_shape()[1].value == 1
        return tf.squeeze(inputs, axis=1)  # N x W x 512

    def _sequence_fn(self, inputs, is_training):
        with tf.variable_scope('sequence_fn'):
            cells_fw = [tf.nn.rnn_cell.BasicLSTMCell(self._hparams.hidden_size, forget_bias=1.0)] * self._hparams.num_layers
            cells_bw = [tf.nn.rnn_cell.BasicLSTMCell(self._hparams.hidden_size, forget_bias=1.0)] * self._hparams.num_layers
            stack_rnn_layer, _, _ = stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, dtype=tf.float32)  # N x length x 512

            if is_training:
                stack_rnn_layer = tf.nn.dropout(stack_rnn_layer, keep_prob=0.5)  # (length, B, 512)

            hidden_size = inputs.get_shape()[2].value
            assert hidden_size == self._hparams.hidden_size * self._hparams.num_layers
            # reshaped = tf.reshape(stack_rnn_layer, [-1, hidden_size])
            weight = tf.get_variable('proj_weight', shape=[hidden_size, self._dparams.num_char_classes],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

            logits = tf.tensordot(stack_rnn_layer, weight, axes=[[2], [0]])

            # logits = tf.argmax(tf.nn.softmax(net), axis=2, name='logits_code')
            logits = tf.transpose(logits, [1, 0, 2], name='time_major_logits')  # time x B x 69

            return logits

    @staticmethod
    def create_init_fn(master_checkpoint):
        """Creates an init operations to restore weights from various checkpoints.

        Args:
          master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.

        Returns:
          a function to run initialization ops.
        """
        all_assign_ops = []
        all_feed_dict = {}

        def assign_from_checkpoint(variables, checkpoint):
            logging.info('Request to re-store %d weights from %s',
                         len(variables), checkpoint)
            if not variables:
                logging.error('Can\'t find any variables to restore.')
                sys.exit(1)

            # Add placeholder to variable assign op, then build feed_dict whose key is placeholder
            # and value is value read from checkpoint.
            assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
            all_assign_ops.append(assign_op)
            all_feed_dict.update(feed_dict)

        if master_checkpoint:
            assign_from_checkpoint(misc.variables_to_restore(), master_checkpoint)

        def init_assign_fn(sess):
            logging.info('Restoring checkpoint(s)')
            sess.run(all_assign_ops, all_feed_dict)

        return init_assign_fn

    def create_eval_ops(self, data, endpoints):
        dense_label = tf.sparse_to_dense(
            data.labels.indices,
            output_shape=[FLAGS.batch_size, self._dparams.max_seq_len],
            sparse_values=data.labels.values,
            default_value=self._dparams.null_code
        )
        dense_prediction = tf.sparse_to_dense(
            sparse_indices=endpoints.decoded[0].indices,
            sparse_values=endpoints.decoded[0].values,
            output_shape=[FLAGS.batch_size, self._hparams.feature_map_seq_len],  # must >= length of endpoints.decoded[0]
            default_value=self._dparams.null_code
        )
        dense_prediction = dense_prediction[:, :self._dparams.max_seq_len]
        # print(dense_prediction)
        # print(dense_label.get_shape().as_list())

        names_to_values = {}
        names_to_updates = {}

        def use_metric(name, value_update_tuple):
            names_to_values[name] = value_update_tuple[0]
            names_to_updates[name] = value_update_tuple[1]

        use_metric(
            'SequenceAccuracy',
            metrics.sequence_accuracy(
                dense_prediction,
                dense_label,
                rej_char=self._dparams.null_code,
                streaming=True,
            )
        )
        for name, value in names_to_values.items():
            summary_name = 'eval/' + name
            # Print info once tensor flow through value node.
            tf.summary.scalar(summary_name, tf.Print(value, [value], summary_name))
        return list(names_to_updates.values()), dense_prediction, dense_label
