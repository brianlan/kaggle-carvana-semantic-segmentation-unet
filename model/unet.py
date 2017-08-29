import tensorflow as tf


from .helpers import conv_layer, max_pooling_layer, up_sampling_layer


class NotSupportedUNetResolution(Exception):
    pass


class UNet:
    lookup_table = {128: 64, 256: 32, 512: 16, 1024: 8}

    def __init__(self, num_classes, input_shape=128, learning_rate=1e-5, tf_scope='unet'):
        if input_shape not in [128, 256, 512, 1024]:
            raise NotSupportedUNetResolution('Only 128, 256, 512 and 1024 are valid shape for UNet, but {!r} is given.'.format(input_shape))

        self.tf_scope = tf_scope
        self.num_classes = num_classes
        self.is_training = tf.placeholder(tf.bool)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.start_num_filters = self.lookup_table[input_shape]
        self.X_train = tf.placeholder(tf.float32, shape=(None, self.input_shape, self.input_shape, 3), name='X_train')
        self.y_train = tf.placeholder(tf.int32, shape=(None, self.input_shape, self.input_shape, 1), name='y_train')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.params = {}
        self.pred = self.loss = self.optimizer = self.train_op = None

    def _build_downward_layer(self, feat, num_filters, d='down'):
        f_shape = (3, 3, num_filters)
        basename = d + str(num_filters)
        self.params[basename+'a'] = conv_layer(feat, f_shape, self.is_training, use_bn=True, name=basename + 'a')
        self.params[basename+'b'] = conv_layer(self.params[basename+'a'], f_shape, self.is_training, use_bn=True, name=basename + 'b')
        self.params[basename+'pool'] = max_pooling_layer(self.params[basename+'b'], 2, 2, name=basename+'pool')
        return self.params[basename+'pool']

    def _build_center_layer(self, feat, num_filters, d='center'):
        f_shape = (3, 3, num_filters)
        basename = d + str(num_filters)
        self.params[basename+'a'] = conv_layer(feat, f_shape, self.is_training, use_bn=True, name=basename + 'a')
        self.params[basename+'b'] = conv_layer(self.params[basename+'a'], f_shape, self.is_training, use_bn=True, name=basename + 'b')
        return self.params[basename+'b']

    def _build_upward_layer(self, feat, num_filters, d='up'):
        up_filter_shape = (3, 3, num_filters)
        basename = d + str(num_filters)
        prev_name = 'down' + str(num_filters) + 'b'

        self.params[basename+'deconv'] = up_sampling_layer(feat, up_filter_shape, name=basename+'deconv')

        merged = tf.concat([self.params[prev_name], self.params[basename+'deconv']], axis=3)
        num_channels_of_merged = self.params[prev_name].shape[3].value + num_filters
        merged.set_shape([None, merged.shape[1].value, merged.shape[2].value, num_channels_of_merged])

        self.params[basename+'a'] = conv_layer(merged, (3, 3, num_filters), self.is_training, use_bn=True, name=basename + 'a')
        self.params[basename+'b'] = conv_layer(self.params[basename+'a'], (3, 3, num_filters), self.is_training, use_bn=True, name=basename + 'b')

        return self.params[basename+'b']

    def build(self):
        with tf.name_scope(self.tf_scope):
            feat = self.X_train
            num_filters = self.start_num_filters

            #################
            #   downward
            #################
            while num_filters < 1024:
                feat = self._build_downward_layer(feat, num_filters)
                num_filters *= 2

            #################
            #   center
            #################
            feat = self._build_center_layer(feat, num_filters)
            num_filters //= 2

            #################
            #   upward
            #################
            while num_filters >= self.start_num_filters:
                feat = self._build_upward_layer(feat, num_filters)
                num_filters //= 2

            #####################
            #   classification
            #####################
            self.pred = conv_layer(feat, (1, 1, self.num_classes), self.is_training, name='classifier')
            # self.flat_y_train = tf.reshape(self.y_train, [-1])
            # self.flat_pred = tf.reshape(self.pred, [-1, self.num_classes])
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.y_train, [-1]),
                                                               logits=tf.reshape(self.pred, [-1, self.num_classes])))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
