import os

from scipy.misc import imread, imresize
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


class ImageReader:
    def __init__(self, data_dir, batch_size=16, as_shape=128, mask_dir=None, file_names=None,
                 random_horizontal_flipper=None, random_hsv_shifter=None):
        """
        If file_names not provided,
        :param data_dir: dir of the data to be read
        :param batch_size: size of a batch
        :param file_names: if file_names not provided, all the data in data_dir will be read
        :param mask_dir: if mask_dir is provided, not only the data, but also the mask (label) will be read
        :param as_shape: will execute image reshape according to provided as_shape param.
        :return: a 2-element tuple (train_data, train_mask), if mask_dir not provided, train_mask will be None
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.as_shape = as_shape
        self.mask_dir = mask_dir
        self.random_horizontal_flipper = random_horizontal_flipper
        self.random_hsv_shifter = random_hsv_shifter
        self.file_names = file_names or [ImageFileName(f.split('.')[0]) for f in os.listdir(data_dir)]
        self.num_total_batches = (len(file_names) - 1) // batch_size + 1
        self.all_img_batches = []
        self.all_mask_batches = []
        self.data_pre_fetched = False

    def read(self):
        def _c(*args):
            return os.path.join(*args)

        def _read_img(data_dir, fname, shape, normalize=False, black_or_white=False):
            path = _c(data_dir, fname)
            img = imread(path)
            img = imresize(img, (shape, shape))

            if normalize:
                img = img // 255 if black_or_white else img / 255

            return img

        if self.data_pre_fetched:
            for cur_batch_idx in range(self.num_total_batches):
                yield self.all_img_batches[cur_batch_idx], self.all_mask_batches[cur_batch_idx]
        else:
            for start in range(0, len(self.file_names), self.batch_size):
                img_batch = []
                mask_batch = []
                end = min(start + self.batch_size, len(self.file_names))
                batch_fnames = self.file_names[start:end]

                for f in batch_fnames:
                    img_batch.append(_read_img(self.data_dir, f.jpg, self.as_shape))

                    if self.mask_dir:
                        im = _read_img(self.mask_dir, f.mask, self.as_shape, normalize=True, black_or_white=True)
                        mask_batch.append(np.expand_dims(im, axis=2))

                img_batch = np.array(img_batch, np.float32)
                mask_batch = np.array(mask_batch, np.float32) if self.mask_dir else None

                if self.random_horizontal_flipper:
                    img_batch, mask_batch = self.random_horizontal_flipper(img_batch, mask_batch)

                if self.random_hsv_shifter:
                    img_batch = self.random_hsv_shifter(img_batch)

                yield img_batch, mask_batch

    def pre_fetch(self):
        if not self.data_pre_fetched:
            r = self.read()
            for img_batch, mask_batch in r:
                self.all_img_batches.append(img_batch)
                self.all_mask_batches.append(mask_batch)

            self.data_pre_fetched = True
