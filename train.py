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
from data_io import read_images, ImageFileName
from image_enhancement import random_horizontal_flip


parser = argparse.ArgumentParser(description='Training phase for Kaggle Carvana Challenge')
parser.add_argument('--model-folder', type=str, required=True, help='the model folder name of training result')
parser.add_argument('--resolution', type=int, choices=[128, 256, 512, 1024], required=True, help='resolution of unet')

args = parser.parse_args()

######################################
#  Global Parameters Definition
######################################
PROJECT_HOME = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet'
CHECKPOINT_DIR = os.path.join(PROJECT_HOME, 'checkpoints')
INPUT_DIR = os.path.join(PROJECT_HOME, 'input')
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(INPUT_DIR, 'train_masks')

EPOCHS_ACCUMULATE_EACH_SAVING = 10
MAX_EPOCH = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
BATCH_SIZE = 16
INPUT_SHAPE = args.resolution
EARLY_STOPPING_PATIENCE = 10

######################################
#  Prepare Train / Validation Data
######################################
df = pd.read_csv(os.path.join(INPUT_DIR, 'train_masks.csv'))
fnames = [ImageFileName(f.split('.')[0]) for f in df['img'].tolist()]
fnames_train, fnames_validation = train_test_split(fnames, test_size=0.2, random_state=233)

cur_checkpoint_path = os.path.join(CHECKPOINT_DIR, args.model_folder)
if not os.path.exists(cur_checkpoint_path):
    os.makedirs(cur_checkpoint_path)

######################################
#  Build Graph and Evaluation
######################################
with tf.Session() as sess:
    unet = UNet(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE)
    unet.build()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    best_val_loss = 9999999
    num_consecutive_worse = 0
    for epoch in range(MAX_EPOCH):
        ##############
        #   Train
        ##############
        start_time = time.time()
        train_data = read_images(TRAIN_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, mask_dir=TRAIN_MASK_DIR,
                                 file_names=fnames_train)
        for batch, (X_batch, y_batch) in enumerate(train_data):
            X_batch, y_batch = random_horizontal_flip(X_batch, y_batch)
            _, loss, pred = sess.run([unet.train_op, unet.loss, unet.pred],
                                     feed_dict={unet.is_training: True, unet.X_train: X_batch, unet.y_train: y_batch})
            logger.info('[epoch {}, batch {}] training loss: {}'.format(epoch, batch, loss))

        logger.info('==== epoch {} took {:.0f} seconds to train. ===='.format(epoch, time.time() - start_time))

        ##########################
        #   Eval Validation set
        ##########################
        start_time = time.time()
        val_data = read_images(TRAIN_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, mask_dir=TRAIN_MASK_DIR,
                               file_names=fnames_validation)
        losses = []
        for batch, (X_batch, y_batch) in enumerate(val_data):
            loss, pred = sess.run([unet.loss, unet.pred],
                                  feed_dict={unet.is_training: False, unet.X_train: X_batch, unet.y_train: y_batch})
            losses.append(loss)

        avg_val_loss = np.average(losses)
        logger.info('==== average validation loss: {} ===='.format(avg_val_loss))
        logger.info('==== epoch {} took {:.0f} seconds to evaluate the validation set. ===='.format(epoch, time.time() - start_time))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            num_consecutive_worse = 0
        else:
            num_consecutive_worse += 1

        def save_checkpoint(sess):
            saver.save(sess, os.path.join(cur_checkpoint_path, 'unet-{}'.format(INPUT_SHAPE)), global_step=epoch)

        if num_consecutive_worse >= EARLY_STOPPING_PATIENCE:
            logger.info('==== Training early stopped because worse val loss lasts for {} epochs. ===='.format(num_consecutive_worse))
            save_checkpoint(sess)
            break

        if (epoch > 0 and epoch % EPOCHS_ACCUMULATE_EACH_SAVING == 0) or epoch == MAX_EPOCH - 1:
            save_checkpoint(sess)

