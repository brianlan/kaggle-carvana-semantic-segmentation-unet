import os

import cv2
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
    def __init__(self, data_dir, batch_size=16, as_shape=128, mask_dir=None, file_names=None, image_augments=None):
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
        self.image_augments = [] if image_augments is None else image_augments
        self.file_names = file_names or [ImageFileName(f.split('.')[0]) for f in os.listdir(data_dir)]
        self.num_total_batches = (len(file_names) - 1) // batch_size + 1
        self.all_img_batches = []
        self.all_mask_batches = []
        self.data_pre_fetched = False

    def read(self, prefetch=False):
        def _c(*args):
            return os.path.join(*args)

        if self.data_pre_fetched:
            for cur_batch_idx in range(self.num_total_batches):
                img_batch, mask_batch = self.all_img_batches[cur_batch_idx], self.all_mask_batches[cur_batch_idx]
                for i in range(img_batch.shape[0]):
                    for aug_func in self.image_augments:
                        img_batch[i, ...], mask_batch[i, ...] = aug_func(img_batch[i, ...], mask_batch[i, ...])

                yield img_batch, mask_batch
        else:
            for start in range(0, len(self.file_names), self.batch_size):
                img_batch = []
                mask_batch = []
                mask = None
                end = min(start + self.batch_size, len(self.file_names))
                batch_fnames = self.file_names[start:end]

                for f in batch_fnames:
                    img = cv2.imread(_c(self.data_dir, f.jpg))
                    img = cv2.resize(img, (self.as_shape, self.as_shape))

                    if self.mask_dir:
                        mask = cv2.imread(_c(self.mask_dir, f.mask), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (self.as_shape, self.as_shape))
                        mask = np.expand_dims(mask, axis=2)

                    if not prefetch:
                        for aug_func in self.image_augments:
                            img, mask = aug_func(img, mask)

                    # cv2.imwrite('/tmp/mine/{}'.format(f.jpg), img)

                    img_batch.append(img)
                    mask_batch.append(mask)

                img_batch = np.array(img_batch, np.float32) / 255
                mask_batch = (np.array(mask_batch, np.float32) / 255) if self.mask_dir else None

                yield img_batch, mask_batch

    def pre_fetch(self):
        if not self.data_pre_fetched:
            r = self.read(prefetch=True)
            for img_batch, mask_batch in r:
                self.all_img_batches.append(img_batch)
                self.all_mask_batches.append(mask_batch)

            self.data_pre_fetched = True
