import scipy
import pandas as pd
from sklearn.model_selection import train_test_split


def read_images(data_dir, file_names=None, mask_dir=None):
    """
    If file_names not provided,
    :param data_dir: dir of the data to be read
    :param file_names: if file_names not provided, all the data in data_dir will be read
    :param mask_dir: if mask_dir is provided, not only the data, but also the mask (label) will be read
    :return:
    """

    # scipy.misc.imread()



file_names = pd.read_csv('~/projects/image-semantic-segmentation/input/train_masks.csv')

pass