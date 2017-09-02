from scipy.misc import imsave, imresize
import numpy as np


def save_image(image: np.ndarray, save_path):
    assert len(image.shape) == 2
    if image.max() == 1:
        image = image * 255
    imsave(save_path, image)


def resize_image(image: np.ndarray, height, width, black_or_white=False):
    resized_image = imresize(image, (height, width), mode='L')

    if black_or_white:
        resized_image = resized_image // 255

    return resized_image


def run_length_encode(mask):
    """ source: https://www.kaggle.com/stainsby/fast-tested-rle
    :param mask: numpy array, 1 - mask, 0 - background
    :return: Returns run length as string formated
    """
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle
