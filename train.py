import os

import scipy
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import INPUT_DIR, TRAIN_DATA_DIR, TRAIN_MASK_DIR


def read_images(data_dir, mask_dir=None, file_names=None):
    """
    If file_names not provided,
    :param data_dir: dir of the data to be read
    :param file_names: if file_names not provided, all the data in data_dir will be read
    :param mask_dir: if mask_dir is provided, not only the data, but also the mask (label) will be read
    :return:
    """

    scipy.misc.imread()


df = pd.read_csv(os.path.join(INPUT_DIR, 'train_masks.csv'))
fnames = [f.split('.')[0] for f in df['img'].tolist()]
fnames_train, fnames_validation = train_test_split(fnames, test_size=0.2, random_state=233)
train_data = read_images(TRAIN_DATA_DIR, mask_dir=TRAIN_MASK_DIR, file_names=fnames_train)

pass
