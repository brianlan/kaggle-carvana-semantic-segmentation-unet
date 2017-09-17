import tensorflow as tf
import cv2
import numpy as np


def calc_dice_loss(y_true, y_pred):
    smooth = tf.constant(1., dtype=tf.float32)
    factor = tf.constant(2., dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    score = (factor * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    loss = 1 - score
    loss = tf.Print(loss, [tf.shape(loss), loss], message='dice_loss: ')
    return loss


def test_main():
    img = cv2.imread('/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/input/train/0495dcf27283_11.jpg')
    y_pred = img[..., :2] / 255
    mask = cv2.imread('/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/input/train_masks/0495dcf27283_11_mask.png', cv2.IMREAD_GRAYSCALE)

    # y_pred = np.argmax(img, axis=2)
    # y_pred = np.expand_dims(img, axis=2)

    y_true = np.expand_dims(mask, axis=2) / 255

    with tf.Session() as sess:
        y_pred_tensor = tf.placeholder(tf.float32, shape=(1280, 1918, 2))
        y_true_tensor = tf.placeholder(tf.int32, shape=(1280, 1918, 1))
        dice_loss_tensor = calc_dice_loss(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1, 2]))
        bce_loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_true_tensor, [-1]),
                                                                         logits=tf.reshape(y_pred_tensor, [-1, 2]))
        init = tf.global_variables_initializer()
        sess.run(init)

        dice_loss, bce_loss = sess.run([dice_loss_tensor, bce_loss_tensor],
                                       feed_dict={y_pred_tensor: y_pred, y_true_tensor: y_true})
        pass
