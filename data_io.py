import os

import scipy
import numpy as np


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
    def mask(self):
        return '{}_mask.png'.format(self.fname)


def read_images(data_dir, batch_size=16, as_shape=128, mask_dir=None, file_names=None):
    """
    If file_names not provided,
    :param data_dir: dir of the data to be read
    :param batch_size: size of a batch
    :param file_names: if file_names not provided, all the data in data_dir will be read
    :param mask_dir: if mask_dir is provided, not only the data, but also the mask (label) will be read
    :param as_shape: will execute image reshape according to provided as_shape param.
    :return: a 2-element tuple (train_data, train_mask), if mask_dir not provided, train_mask will be None
    """

    def _c(*args):
        return os.path.join(*args)

    def _read_img(data_dir, fname, shape, normalize=False, black_or_white=False):
        path = _c(data_dir, fname)
        img = scipy.misc.imread(path)
        img = scipy.misc.imresize(img, (shape, shape))

        if normalize:
            img = img // 255 if black_or_white else img / 255

        return img

    file_names = file_names or [ImageFileName(f.split('.')[0]) for f in os.listdir(data_dir)]
    for start in range(0, len(file_names), batch_size):
        img_batch = []
        mask_batch = []
        end = min(start + batch_size, len(file_names))
        batch_fnames = file_names[start:end]

        for f in batch_fnames:
            img_batch.append(_read_img(data_dir, f.jpg, as_shape))

            if mask_dir:
                im = _read_img(mask_dir, f.mask, as_shape, normalize=True, black_or_white=True)
                mask_batch.append(np.expand_dims(im, axis=2))

        yield np.array(img_batch, np.float32), np.array(mask_batch, np.float32) if mask_dir else None
