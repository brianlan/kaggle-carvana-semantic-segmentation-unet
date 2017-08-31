import numpy as np

from image_op import run_length_encode, rlencode


def test_run_length_encode():
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

    encoded = run_length_encode(image)

    assert encoded == [[12, 1],
                       [19, 4],
                       [27, 3],
                       [35, 3],
                       [43, 4],
                       [52, 1]]
