import os

import pandas as pd
from sklearn.model_selection import train_test_split

from settings import INPUT_DIR, TRAIN_DATA_DIR, TRAIN_MASK_DIR
from data_io import read_images, ImageFileName


df = pd.read_csv(os.path.join(INPUT_DIR, 'train_masks.csv'))
fnames = [ImageFileName(f.split('.')[0]) for f in df['img'].tolist()]
fnames_train, fnames_validation = train_test_split(fnames, test_size=0.2, random_state=233)
train_data = read_images(TRAIN_DATA_DIR, batch_size=16, as_shape=(128, 128), mask_dir=TRAIN_MASK_DIR,
                         file_names=fnames_train)

pass
