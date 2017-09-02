import os
import time
import argparse

import tensorflow as tf
import pandas as pd
import numpy as np

from data_io import read_images, ImageFileName
from model.unet import UNet
from image_op import resize_image, run_length_encode
from logger import logger


parser = argparse.ArgumentParser(description='Generate Submissions for Kaggle Carvana Challenge')
parser.add_argument('--model-folder', type=str, required=True, help='the model folder name of training result')
parser.add_argument('--resolution', type=int, choices=[128, 256, 512, 1024], required=True, help='resolution of unet')

args = parser.parse_args()

PROJECT_HOME = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet'
INPUT_DIR = os.path.join(PROJECT_HOME, 'input')
SUBMISSION_OUTPUT_DIR = os.path.join(PROJECT_HOME, 'output')
MODEL_DIR = os.path.join(PROJECT_HOME, 'checkpoints', args.model_folder)
TEST_DATA_DIR = os.path.join(INPUT_DIR, 'test')
ORIGINAL_IMAGE_SIZE = (1280, 1918)
INPUT_SHAPE = args.resolution
BATCH_SIZE = 16
NUM_CLASSES = 2

sample_submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission_mvp.csv'))
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
    start = batch_start = time.time()
    for batch, (X_batch, _) in enumerate(test_data):
        pred, = sess.run([unet.pred], feed_dict={unet.is_training: False, unet.X_train: X_batch})
        for i in range(pred.shape[0]):
            img = resize_image(np.argmax(pred[i, :, :, :], axis=2), *ORIGINAL_IMAGE_SIZE, black_or_white=True)
            sample_submission.iloc[batch * BATCH_SIZE + i, ]['rle_mask'] = run_length_encode(img)

        logger.info('[batch {}] took {:.2f}s to eval. Total elapsed time: {:.2f}s'.format(batch,
                                                                                          time.time() - batch_start,
                                                                                          time.time() - start))
        batch_start = time.time()

    sample_submission.to_csv(os.path.join(SUBMISSION_OUTPUT_DIR, 'submission_{}.csv.gz'.format(args.model_folder)),
                             index=False,
                             compression='gzip')
