import tensorflow as tf


def test_conv2d():
    tf.set_random_seed(1)
    x = tf.random_normal(shape=[1, 5, 5, 3])
    kernel = tf.random_normal(shape=[2, 2, 3, 1])
    y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    a = y.eval(session=sess)

    pass


def test_conv2d_transpose():
    tf.set_random_seed(1)
    x = tf.random_normal(shape=[1, 3, 3, 1])
    kernel = tf.random_normal(shape=[2, 2, 3, 1])
    y = tf.nn.conv2d_transpose(x, kernel, output_shape=[1, 5, 5, 3],
                               strides=[1, 2, 2, 1], padding="SAME")
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    a = y.eval(session=sess)

    pass
