import numpy as np
from scipy.misc import imread, imshow, imsave, imresize
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def random_hsv_shift(image, hue_shift, sat_shift, val_shift):
    hsv = rgb_to_hsv(image / 255.0)
    hsv[:, :, 0] += hue_shift
    hsv[:, :, 1] += sat_shift
    hsv[:, :, 2] += val_shift
    hsv[hsv > 1] = 1
    hsv[hsv < 0] = 0
    image = hsv_to_rgb(hsv)

    return image


def randomHueSaturationValue(image, hue_shift, sat_shift, val_shift):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    h = cv2.add(h, hue_shift)
    s = cv2.add(s, sat_shift)
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def test_hsv_shift():
    img_path = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/input/train/0ce66b539f52_11.jpg'
    hue_shift = np.random.uniform(-50, 50)
    sat_shift = np.random.uniform(-5, 5)
    val_shift = np.random.uniform(-15, 15)

    img = imread(img_path)
    imsave('/tmp/raw.png', img)
    img2 = random_hsv_shift(img, hue_shift / 180.0, sat_shift / 255.0, val_shift / 255.0)
    imsave('/tmp/hsv1.png', img2)

    import cv2
    cv_img = cv2.imread(img_path)
    cv_img2 = randomHueSaturationValue(cv_img, hue_shift, sat_shift, val_shift)
    cv2.imwrite('/tmp/hsv2.png', cv_img2)


# def test_hsv_shift2():
#     with open('/tmp/cv2_random_hsv/cv2_random_hsv.txt', 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             img_name, hue_shift, sat_shift, val_shift = line.split(',')
#             img = imread('/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/input/train/{}.jpg'.format(img_name))
#             img = imresize(img, (128, 128))
#             img2 = random_hsv_shift(img, float(hue_shift) / 180.0, float(sat_shift) / 255.0, float(val_shift) / 255.0)
#             imsave('/tmp/matplotlib_random_hsv/{}_rlan.png'.format(img_name), img2)


# test_hsv_shift2()