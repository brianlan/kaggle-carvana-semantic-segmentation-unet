import os
import time
import argparse

import scipy
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from logger import logger
from model.unet import UNet
from data_io import ImageFileName, ImageReader
from image_augment import random_horizontal_flip, random_hsv_shift, random_shift_scale_rotate
from utils import store_true


parser = argparse.ArgumentParser(description='Training phase for Kaggle Carvana Challenge')
parser.add_argument('--model-folder', type=str, required=True, help='the model folder name of training result')
parser.add_argument('--resolution', type=int, choices=[128, 256, 512, 1024], required=True, help='resolution of unet')
parser.add_argument('--batch-size', type=int, required=True, help='batch size')
parser.add_argument('--image-prefetch', dest='image_prefetch', default=False, action="store_true", help='whether prefetch data into memory.')

args = parser.parse_args()

######################################
#  Global Parameters Definition
######################################
PROJECT_HOME = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet'
CHECKPOINT_DIR = os.path.join(PROJECT_HOME, 'checkpoints')
INPUT_DIR = os.path.join(PROJECT_HOME, 'input')
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(INPUT_DIR, 'train_masks')

SAVING_INTERVAL = 10
MAX_EPOCH = 100
NUM_CLASSES = 2
BATCH_SIZE = args.batch_size
INPUT_SHAPE = args.resolution
EARLY_STOPPING_PATIENCE = 8
LR_REDUCE_PATIENCE = 4
LR_REDUCE_FACTOR = 0.1

LEARNING_RATE_SETTINGS = [
    {'max_epoch': 20, 'lr': 0.001},
    {'max_epoch': 70, 'lr': 0.0001},
    {'max_epoch': 40, 'lr': 0.00001},
    {'max_epoch': 40, 'lr': 0.000001},
]

######################################
#  Prepare Train / Validation Data
######################################
df = pd.read_csv(os.path.join(INPUT_DIR, 'train_masks.csv'))
fnames = [ImageFileName(f.split('.')[0]) for f in df['img'].tolist()]
fnames_train, fnames_validation = train_test_split(fnames, test_size=0.2, random_state=233)


def random_hsv_shifter(image, mask, u=0.5):
    return random_hsv_shift(image, mask, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15), u=u)


def random_shift_scale_rotate_operator(image, mask, u=0.5):
    return random_shift_scale_rotate(image, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0), u=u)


train_img_reader = ImageReader(TRAIN_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, mask_dir=TRAIN_MASK_DIR,
                               file_names=fnames_train,
                               image_augments=[random_hsv_shifter, random_shift_scale_rotate_operator,
                                               random_horizontal_flip])
val_img_reader = ImageReader(TRAIN_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, mask_dir=TRAIN_MASK_DIR,
                             file_names=fnames_validation)

if args.image_prefetch:
    t0 = time.time()
    train_img_reader.pre_fetch()
    logger.info('==== Training data pre-fetch took {:.2f}s. ===='.format(time.time() - t0))

    t0 = time.time()
    val_img_reader.pre_fetch()
    logger.info('==== Validation data pre-fetch took {:.2f}s. ===='.format(time.time() - t0))

######################################
#  Build Graph and Evaluation
######################################
cur_checkpoint_path = os.path.join(CHECKPOINT_DIR, args.model_folder)
if not os.path.exists(cur_checkpoint_path):
    os.makedirs(cur_checkpoint_path)


def main():
    with tf.Session() as sess:
        unet = UNet(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
        unet.build()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        best_val_loss = 9999999
        num_consec_worse_earlystop = 0
        num_consec_worse_lr = 0
        # learning_rate = 1e-4

        for s, lrs in enumerate(LEARNING_RATE_SETTINGS):
            for epoch in range(lrs['max_epoch']):
                ##############
                #   Train
                ##############
                start_time = time.time()
                train_data = train_img_reader.read()
                for batch, (X_batch, y_batch) in enumerate(train_data):
                    _, loss, pred = sess.run([unet.train_op, unet.loss, unet.pred],
                                             feed_dict={unet.is_training: True, unet.X_train: X_batch,
                                                        unet.y_train: y_batch, unet.learning_rate: lrs['lr']})
                    logger.info('[set {}, epoch {}, batch {}] training loss: {}'.format(s, epoch, batch, loss))

                logger.info('==== set {}, epoch {} took {:.0f} seconds to train. ===='.format(s, epoch, time.time() - start_time))

                ##########################
                #   Eval Validation set
                ##########################
                start_time = time.time()
                val_data = val_img_reader.read()
                losses = []
                for batch, (X_batch, y_batch) in enumerate(val_data):
                    loss, pred = sess.run([unet.loss, unet.pred],
                                          feed_dict={unet.is_training: False, unet.X_train: X_batch, unet.y_train: y_batch})
                    losses.append(loss)

                avg_val_loss = np.average(losses)
                logger.info('==== average validation loss: {} ===='.format(avg_val_loss))
                logger.info('==== set {}, epoch {} took {:.0f} seconds to evaluate the validation set. ===='.format(s, epoch, time.time() - start_time))

                def save_checkpoint(sess):
                    saver.save(sess, os.path.join(cur_checkpoint_path, 'unet-{}'.format(INPUT_SHAPE)), global_step=s*len(LEARNING_RATE_SETTINGS)+epoch)

                if lrs.get('reduce_factor'):
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        # num_consec_worse_earlystop = 0
                        num_consec_worse_lr = 0
                    else:
                        # num_consec_worse_earlystop += 1
                        num_consec_worse_lr += 1

                    if num_consec_worse_lr >= lrs.get('reduce_patience'):
                        lrs['lr'] *= lrs.get('reduce_factor')
                        logger.info('==== val loss did not improve for {} epochs, learning rate reduced to {}. ===='.format(
                            num_consec_worse_lr, lrs['lr']))
                        num_consec_worse_lr = 0

                # if num_consec_worse_earlystop >= EARLY_STOPPING_PATIENCE:
                #     logger.info('==== Training early stopped because worse val loss lasts for {} epochs. ===='.format(num_consec_worse_earlystop))
                #     save_checkpoint(sess)
                #     break

                if (epoch > 0 and epoch % SAVING_INTERVAL == 0) or epoch == lrs['max_epoch'] - 1:
                    save_checkpoint(sess)


main()
