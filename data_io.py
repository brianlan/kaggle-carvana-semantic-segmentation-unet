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
    def mask(self):
        return '{}_mask.png'.format(self.fname)


def read_images(data_dir, batch_size=16, as_shape=(128, 128), mask_dir=None, file_names=None):
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

    def _append_img(image_list, data_dir, fname, shape):
        path = _c(data_dir, fname)
        img = scipy.misc.imread(path)
        img = scipy.misc.imresize(img, shape)
        image_list.append(img)

    file_names = file_names or [ImageFileName(f.split('.')[0]) for f in os.listdir(data_dir)]
    while True:
        for start in range(0, len(file_names), batch_size):
            img_batch = []
            mask_batch = []
            end = min(start + batch_size, len(file_names))
            batch_fnames = file_names[start:end]

            for f in batch_fnames:
                _append_img(img_batch, data_dir, f.jpg, as_shape)

                if mask_dir:
                    _append_img(mask_batch, mask_dir, f.mask, as_shape)

            yield img_batch, mask_batch if mask_dir else None
