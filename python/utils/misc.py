# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to support building models for StreetView text transcription."""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import re
import logging


def logits_to_log_prob(logits):
    """Computes log probabilities using numerically stable trick.

    This uses two numerical stability tricks:
    1) softmax(x) = softmax(x - c) where c is a constant applied to all
    arguments. If we set c = max(x) then the softmax is more numerically
    stable.
    2) log softmax(x) is not numerically stable, but we can stabilize it
    by using the identity log softmax(x) = x - log sum exp(x)

    Args:
      logits: Tensor of arbitrary shape whose last dimension contains logits.

    Returns:
      A tensor of the same shape as the input, but with corresponding log
      probabilities.
    """

    with tf.variable_scope('log_probabilities'):
        reduction_indices = len(logits.shape.as_list()) - 1
        # keep_dims: Deprecated alias for `keepdims`.
        # reduction_indices: The old (deprecated) name for axis.
        max_logits = tf.reduce_max(
            logits, axis=reduction_indices, keepdims=True)
        safe_logits = tf.subtract(logits, max_logits)
        sum_exp = tf.reduce_sum(
            tf.exp(safe_logits),
            axis=reduction_indices,
            keepdims=True)
        log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
    return log_probs


def variables_to_restore(scope=None, strip_scope=False):
    """Returns a list of variables to restore for the specified list of methods.

    It is supposed that variable name starts with the method's scope (a prefix
    returned by _method_scope function).

    Args:
      methods_names: a list of names of configurable methods. # 木有这个参数
      strip_scope: if True will return variable names without method's scope.
        If methods_names is None will return names unchanged.
      model_scope: a scope for a whole model.

    Returns:
      a dictionary mapping variable names to variables for restore.
    """
    if scope:
        variable_map = {}
        method_variables = slim.get_variables_to_restore(include=[scope])
        for var in method_variables:
            if strip_scope:
                var_name = var.op.name[len(scope) + 1:]
            else:
                var_name = var.op.name
            variable_map[var_name] = var

        return variable_map
    else:
        return {v.op.name: v for v in slim.get_variables_to_restore()}


def reverse_dict(m_dict):
    return dict(zip(m_dict.values(), m_dict.keys()))


def read_char_vocab(filename, null_character=u'\u2591'):
    """Get a dictionary whose key is char and value is code."""
    pattern = re.compile(r'(\d+)\t(.+)')
    char_vocab = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                if i == 0:
                    # No need to set code of ' ' to 0, because one-hot of 0 is zero tensor.
                    # We can directly set first rnn input to zero tensor.
                    continue
                logging.warning('incorrect vocab file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)
            if char == '<nul>':
                char = null_character
            char_vocab[char] = code
    return char_vocab


def read_code_vocab(filename, null_character=u'\u2591'):
    pattern = re.compile(r'(\d+)\t(.+)')
    code_vocab = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
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


def _dict_to_array(id_to_char, default_character):
    num_char_classes = max(id_to_char.keys()) + 1
    array = [default_character] * num_char_classes
    for k, v in id_to_char.items():
        array[k] = v
    return array


def augment_image(image):
    distorted_image = distort_color(image, np.random.randint(4))
    distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)
    return distorted_image


def distort_color(image, color_ordering=0, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)
