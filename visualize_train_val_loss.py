import subprocess

import pandas as pd


log_path = "/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/log/20170829"
output_path = "/tmp/train_val_error.csv"

train_output = subprocess.check_output("cat {} | grep 'batch 254' | awk '{{print $14}}'".format(log_path), shell=True)
val_output = subprocess.check_output("cat {} | grep 'validation error' | awk '{{print $12}}'".format(log_path), shell=True)
train_err = train_output.strip().split(b'\n')
val_err = val_output.strip().split(b'\n')

df = pd.DataFrame({'train': train_err, 'val': val_err})
df['train'] = df['train'].astype(float)
df['val'] = df['val'].astype(float)
df.to_csv(output_path, index=False)
