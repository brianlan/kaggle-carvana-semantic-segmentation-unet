import tensorflow as tf


from .helpers import conv_layer, max_pooling_layer, up_sampling_layer


class NotSupportedUNetResolution(Exception):
    pass


class UNet:
    lookup_table = {128: 64, 256: 32, 512: 16, 1024: 8}

    def __init__(self, num_classes, input_shape=128, tf_scope='unet'):
        if input_shape not in [128, 256, 512, 1024]:
            raise NotSupportedUNetResolution('Only 128, 256, 512 and 1024 are valid shape for UNet, but {!r} is given.'.format(input_shape))

        self.tf_scope = tf_scope
        self.num_classes = num_classes
        self.input_shape = (input_shape, input_shape)
        self.start_num_filters = self.lookup_table[input_shape]
        self.params = {
            'down': [],
            'center': [],
            'up': []
        }

    def _build_downward_layer(self, feat, num_filters, d='down'):
        f_shape = (3, 3, num_filters)
        basename = d + num_filters
        self.params[basename+'a'] = conv_layer(feat, f_shape, 1, False, use_bn=True, name=basename+'a')
        self.params[basename+'b'] = conv_layer(self.params[d][-1], f_shape, 1, False, use_bn=True, name=basename+'b')
        self.params[basename+'pool'] = max_pooling_layer(self.params[d][-1], 2, 2, name=basename+'pool')
        return self.params[basename+'pool']

    def _build_upward_layer(self, feat, num_filters, d='up'):
        up_filter_shape = (2, 2, num_filters)
        basename = d + num_filters
        self.params[basename+'deconv'] = up_sampling_layer(feat, up_filter_shape, (feat.shape[1], feat.shape[2]), stride=2, name=basename+'deconv')
        merged = tf.concat([self.param['down'+num_filters+'b'], self.params[basename+'deconv']], axis=3)
        self.params[basename+'a'] = conv_layer(merged, (3, 3, num_filters), 1, False, use_bn=True, name=basename+'a')
        self.params[basename+'b'] = conv_layer(self.params[basename+'a'], (3, 3, num_filters), 1, False, use_bn=True, name=basename+'b')
        return self.params[basename+'b']

    def build(self):
        with tf.name_scope(self.tf_scope):
            feat = tf.placeholder(tf.float32, shape=(self.input_shape, self.input_shape, 3), name='X_train')
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
            feat = self._build_downward_layer(feat, num_filters, d='center')
            num_filters /= 2

            #################
            #   upward
            #################
            while num_filters >= self.start_num_filters:
                feat = self._build_upward_layer(feat, num_filters)
                num_filters /= 2
