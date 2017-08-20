import os

import scipy


class ImageFileName(str):
    def __init__(self, fname):
        self.fname = fname

    def __str__(self):
        return self.fname

    def __repr__(self):
        return self.fname

    @property
    def jpg(self):
        return '{}.jpg'.format(self.fname)

    @property
    def png(self):
        return '{}.png'.format(self.fname)


def read_images(data_dir, mask_dir=None, file_names=None, as_shape=(128, 128)):
    """
    If file_names not provided,
    :param data_dir: dir of the data to be read
    :param file_names: if file_names not provided, all the data in data_dir will be read
    :param mask_dir: if mask_dir is provided, not only the data, but also the mask (label) will be read
    :param as_shape: will execute image reshape according to provided as_shape param.
    :return: a 2-element tuple (train_data, train_mask), if mask_dir not provided, train_mask will be None
    """

    def _c(*args):
        return os.path.join(*args)

    for f in file_names:
        img_path = _c(data_dir, f.jpg)
        img = scipy.misc.imread(img_path)
        img = scipy.misc.imresize(img, as_shape)
        pass

