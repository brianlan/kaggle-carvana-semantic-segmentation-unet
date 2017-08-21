import tensorflow as tf


from .helpers import conv_layer


class UNet:
    def __init__(self, num_classes, input_shape=128, tf_scope='unet'):
        self.tf_scope = tf_scope
        self.num_classes = num_classes
        self.input_shape = (input_shape, input_shape)

    def build(self):
        with tf.name_scope(self.tf_scope):
            X_train = tf.placeholder(tf.float32, shape=(self.input_shape, self.input_shape, 3), name='X_train')
            self.conv1 = conv_layer(X_train, (3, 3, 64))
