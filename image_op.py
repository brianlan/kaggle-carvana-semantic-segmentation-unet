from scipy.misc import imsave
import numpy as np


def save_image(image: np.ndarray, save_path):
    assert len(image.shape) == 2
    if image.max() == 1:
        image = image * 255
    imsave(save_path, image)
