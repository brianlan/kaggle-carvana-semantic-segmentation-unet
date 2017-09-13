import numpy as np
import cv2


#####################################################################################################
# Below 3 functions are sourcing from below link with minor change.
#     REFERENCE: https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523
#     REFERENCE: https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
#####################################################################################################
def random_horizontal_flip(images, masks, u=0.5):
    for i in range(images.shape[0]):
        if np.random.random() < u:
            images[i, :, :, :] = cv2.flip(images[i, :, :, :], 1)
            masks[i, :, :, 0] = cv2.flip(masks[i, :, :, 0], 1)

    return images, masks


def random_hsv_shift(images, masks, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255),
                     u=0.5):
    for i in range(images.shape[0]):
        if np.random.random() < u:
            img = cv2.cvtColor(images[i, :, :, :], cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img)
            hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            img = cv2.merge((h, s, v))
            images[i, :, :, :] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return images, masks


def random_shift_scale_rotate(images, masks, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1),
                              rotate_limit=(-45, 45), aspect_limit=(0, 0), borderMode=cv2.BORDER_CONSTANT, u=0.5):
    for i in range(images.shape[0]):
        if np.random.random() < u:
            height, width, channel = images[i, :, :, :].shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            images[i, :, :, :] = cv2.warpPerspective(images[i, :, :, :], mat, (width, height), flags=cv2.INTER_LINEAR,
                                                     borderMode=borderMode, borderValue=(0, 0, 0,))
            masks[i, :, :, 0] = cv2.warpPerspective(masks[i, :, :, 0], mat, (width, height), flags=cv2.INTER_LINEAR,
                                                    borderMode=borderMode, borderValue=(0, 0, 0,))

    return images, masks

