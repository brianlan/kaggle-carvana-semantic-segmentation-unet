import tensorflow as tf


def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batchnorm_layer(X, is_test, offset, scale, iteration, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration)
    if convolutional:
        mean, variance = tf.nn.moments(X, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(X, [0])

    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    bn = tf.nn.batch_normalization(X, m, v, offset, scale, variance_epsilon=1e-5)
    return bn, update_moving_averages


def conv_layer(X, filter_shape, name):
    """
    :param X: the input data, which is of shape (N, H, W, C)
    :param filter_shape: the shape of filter, which is of shape (H, W, C)
    :param name: tf's variable scope name
    :return:
    """
    with tf.variable_scope(name) as scope:
        filt = weight_variable(filter_shape)
        conv = tf.nn.conv2d(X, filt, [1, 1, 1, 1], padding='SAME')
        bias = bias_variable(filter_shape[2])
        bn = batchnorm_layer(conv + bias)
        relu = tf.nn.relu(bn)

        # Add summary to Tensorboard
        activation_summary(relu)
        return relu
