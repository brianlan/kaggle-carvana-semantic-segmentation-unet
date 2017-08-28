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


def batchnorm_layer(X, is_test, offset, scale, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999)
    if convolutional:
        mean, variance = tf.nn.moments(X, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(X, [0])

    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    bn = tf.nn.batch_normalization(X, m, v, offset, scale, variance_epsilon=1e-5)
    return bn, update_moving_averages


def conv_layer(X, filter_shape, is_test, stride=1, use_bn=False, name=None):
    """
    :param X: the input data, which is of shape (N, H, W, C)
    :param filter_shape: the shape of filter, which is of shape (H, W, C)
    :param stride:
    :param is_test: as the var name presents
    :param use_bn: boolean param that indicates whether the conv_layer uses batchnorm whithin it
    :param name: tf's variable scope name
    :return:
    """
    with tf.variable_scope(name) as scope:
        filt = weight_variable(tf.TensorShape([filter_shape[0], filter_shape[1], X.shape[3].value, filter_shape[2]]))
        conv = tf.nn.conv2d(X, filt, [1, stride, stride, 1], padding='SAME')
        bias = bias_variable([filter_shape[2]])
        logits = conv + bias

        if use_bn:
            scale = tf.Variable(tf.ones([filter_shape[2]]))  # aka. alpha
            offset = tf.Variable(tf.zeros([filter_shape[2]]))  # aka. beta
            bn, _ = batchnorm_layer(logits, is_test, offset, scale, convolutional=True)

        relu = tf.nn.relu(bn if use_bn else logits)

        # Add summary to Tensorboard
        # activation_summary(relu)
        return relu


def max_pooling_layer(X, kernel_size=2, stride=2, padding='SAME', name=None):
    """input data should be of shape (N, H, W, C)"""
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(X, ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding, name=name)
    return pool


def up_sampling_layer(X, filter_shape, stride=2, padding='SAME', name=None):
    with tf.variable_scope(name):
        filt = weight_variable((filter_shape[0], filter_shape[1], filter_shape[2], X.shape[3].value))
        X_shape = tf.shape(X)
        output_shape = tf.stack([X_shape[0], X_shape[1] * 2, X_shape[2] * 2, filter_shape[2]])
        deconv = tf.nn.conv2d_transpose(X, filt, output_shape, strides=[1, stride, stride, 1], padding=padding)

    return deconv