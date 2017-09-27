import subprocess
import argparse

import pandas as pd


parser = argparse.ArgumentParser(description='visualize the train / val losses')
parser.add_argument('--log-path', type=str, required=True, help='the path to the log')
args = parser.parse_args()

log_path = args.log_path  #"/home/rlan/projects/Kaggle/Carnava/kaggle-carvana-semantic-segmentation-unet/log/20170915"
output_path = "/tmp/train_val_error.csv"

train_output = subprocess.check_output("cat {} | grep 'batch 508] training loss' | awk '{{print $16}}'".format(log_path), shell=True)
val_output = subprocess.check_output("cat {} | grep 'validation loss' | awk '{{print $12}}'".format(log_path), shell=True)
train_err = train_output.strip().split(b'\n')
val_err = val_output.strip().split(b'\n')

df = pd.DataFrame({'train': train_err, 'val': val_err})
df['train'] = df['train'].astype(float)
df['val'] = df['val'].astype(float)
df.to_csv(output_path, index=False)
