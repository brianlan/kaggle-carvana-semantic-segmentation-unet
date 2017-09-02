from scipy.misc import imsave, imresize
import numpy as np


def save_image(image: np.ndarray, save_path):
    assert len(image.shape) == 2
    if image.max() == 1:
        image = image * 255
    imsave(save_path, image)


def resize_image(image: np.ndarray, height, width):
    return imresize(image, (height, width), mode='L')


def run_length_encode(image: np.ndarray):
    vector = image.T.reshape([image.shape[0] * image.shape[1]])
    starts, lengths, values = rlencode(vector)
    return np.stack((starts[values > 0]+1, lengths[values > 0]), axis=1).tolist()


def flat(encoded):
    return ' '.join([str(j) for i in encoded for j in i])


def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values
