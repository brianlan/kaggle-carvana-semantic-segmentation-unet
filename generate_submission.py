import os
import time
import argparse

import tensorflow as tf
import pandas as pd
import numpy as np

from data_io import read_images, ImageFileName
from model.unet import UNet
from image_op import save_image, resize_image, run_length_encode
from logger import logger


PROJECT_HOME = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet'
INPUT_DIR = os.path.join(PROJECT_HOME, 'input')
SUBMISSION_OUTPUT_DIR = os.path.join(PROJECT_HOME, 'output')
MODEL_DIR = os.path.join(PROJECT_HOME, 'checkpoints', '1504019330')
TEST_DATA_DIR = os.path.join(INPUT_DIR, 'test')
ORIGINAL_IMAGE_SIZE = (1280, 1918)
INPUT_SHAPE = 512
BATCH_SIZE = 16
NUM_CLASSES = 2

sample_submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
fnames = [ImageFileName(f.split('.')[0]) for f in sample_submission['img'].tolist()]
test_data = read_images(TEST_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, file_names=fnames)

with tf.Session() as sess:
    unet = UNet(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    unet.build()
    model = tf.train.get_checkpoint_state(MODEL_DIR)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    saver.restore(sess, model.model_checkpoint_path)

    ##########################
    #   Eval Validation set
    ##########################
    start_time = time.time()
    for batch, (X_batch, _) in enumerate(test_data):
        pred, = sess.run([unet.pred], feed_dict={unet.is_training: False, unet.X_train: X_batch})
        for i in range(pred.shape[0]):
            img = resize_image(np.argmax(pred[i, :, :, :], axis=2), *ORIGINAL_IMAGE_SIZE)
            encoded = run_length_encode(img)
        # last_image = resize_image(np.argmax(pred[pred.shape[0] - 1, :, :, :], axis=2) * 255, *ORIGINAL_IMAGE_SIZE)
        # save_image(last_image, os.path.join(PROJECT_HOME, 'sample_results', 'test', 'batch_{}.png'.format(batch)))
        logger.info('[batch {}] took {:.0f} seconds to evaluate.'.format(batch, time.time() - start_time))

pass
