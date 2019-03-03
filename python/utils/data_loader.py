import collections
import tensorflow as tf

InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'labels'])

# A namedtuple to define a configuration for shuffled batch fetching.
#   num_batching_threads: A number of parallel threads to fetch data.
#   queue_capacity: a max number of elements in the batch shuffling queue.
#   min_after_dequeue: a min number elements in the queue after a dequeue, used
#     to ensure a level of mixing of elements.
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def central_crop(image, crop_size):
    """Returns a central crop for the specified size of an image.

    Args:
      image: A tensor with shape [height, width, channels]
      crop_size: A tuple (crop_width, crop_height)

    Returns:
      A tensor of shape [crop_height, crop_width, channels].
    """
    with tf.variable_scope('CentralCrop'):
        target_width, target_height = crop_size
        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        assert_op1 = tf.Assert(
            tf.greater_equal(image_height, target_height),
            ['image_height < target_height', image_height, target_height])
        assert_op2 = tf.Assert(
            tf.greater_equal(image_width, target_width),
            ['image_width < target_width', image_width, target_width])
        with tf.control_dependencies([assert_op1, assert_op2]):
            offset_width = (image_width - target_width) / 2
            offset_height = (image_height - target_height) / 2
            return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                                 target_height, target_width)


def get_data(dataset):
    iterator = dataset.dataset.make_one_shot_iterator()
    images, labels, _ = iterator.get_next()

    return InputEndpoints(
        images=images,
        labels=labels)
