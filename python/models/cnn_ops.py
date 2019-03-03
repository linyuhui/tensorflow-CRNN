import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers


def tf_conv2d(inputs, out_channels, kernel_size, padding='SAME',
           stride=1, weight_initializer=None, bias_initializer=None,
           groups=1, use_bias=True, data_format='NHWC', name=None):
    with tf.variable_scope(name):
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channels = in_shape[channel_axis]
        assert in_channels is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        padding = padding.upper()

        if isinstance(kernel_size, list):
            kernel_size = [kernel_size[0], kernel_size[1], in_channels / groups, out_channels]
        else:
            kernel_size = [kernel_size, kernel_size, in_channels / groups, out_channels]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        if weight_initializer is None:
            weight_initializer = layers.initializers.variance_scaling_initializer()
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer()

        weight = tf.get_variable('weight', kernel_size, initializer=weight_initializer)
        bias = None

        if use_bias:
            bias = tf.get_variable('bias', [out_channels], initializer=bias_initializer)

        if groups == 1:
            net = tf.nn.conv2d(inputs, weight, strides, padding, data_format=data_format)
        else:
            inputs = tf.split(inputs, groups, channel_axis)
            kernels = tf.split(weight, groups, 3)
            outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format) for i, k in zip(inputs, kernels)]
            net = tf.concat(outputs, channel_axis)

        net = (tf.nn.bias_add(net, bias, data_format=data_format) if use_bias else net)

    return net


def tf_max_pooling2d(inputs, kernel_size, stride=None, padding='VALID', data_format='NHWC', name=None):
    padding = padding.upper()

    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, list):
        kernel_size = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else [1, 1, kernel_size[0], kernel_size[1]]
    else:
        kernel_size = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' else [1, 1, kernel_size, kernel_size]

    if isinstance(stride, list):
        strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' else [1, 1, stride[0], stride[1]]
    else:
        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

    return tf.nn.max_pool(value=inputs, ksize=kernel_size, strides=strides, padding=padding,
                          data_format=data_format, name=name)

def batch_norm(inputs, is_training):
    return layers.batch_norm(inputs, scale=True, is_training=is_training)
