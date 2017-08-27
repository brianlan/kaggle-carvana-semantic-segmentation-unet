import os
import time

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from model.unet import UNet
from data_io import read_images, ImageFileName


######################################
#  Global Parameters Definition
######################################
PROJECT_HOME = '/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet'
INPUT_DIR = os.path.join(PROJECT_HOME, 'input')
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(INPUT_DIR, 'train_masks')

EPOCHS = 50
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
BATCH_SIZE = 16
INPUT_SHAPE = 128

######################################
#  Prepare Train / Validation Data
######################################
df = pd.read_csv(os.path.join(INPUT_DIR, 'train_masks.csv'))
fnames = [ImageFileName(f.split('.')[0]) for f in df['img'].tolist()]
fnames_train, fnames_validation = train_test_split(fnames, test_size=0.2, random_state=233)
train_data = read_images(TRAIN_DATA_DIR, batch_size=BATCH_SIZE, as_shape=INPUT_SHAPE, mask_dir=TRAIN_MASK_DIR,
                         file_names=fnames_train)

######################################
#  Build Graph and Evaluation
######################################
with tf.Session() as sess:
    unet = UNet(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, learning_rate=LEARNING_RATE)
    unet.build()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(EPOCHS):
        start_time = time.time()
        for batch, (X_batch, y_batch) in enumerate(train_data):
            _, loss, pred = sess.run([unet.train_op, unet.cross_entropy, unet.pred],
                                     feed_dict={unet.X_train: X_batch, unet.y_train: y_batch})

            print('[epoch {}, batch {}] cross_entropy: {}'.format(epoch, batch, loss))

        print('epoch {} took {:.0f} seconds to train.'.format(epoch, time.time() - start_time))

    pass
