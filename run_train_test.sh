#!/bin/bash

unixts=$(date +%s)
resolution=128
batch_size=16

python train.py --model-folder ${unixts} --resolution ${resolution} --batch-size ${batch_size} --image-prefetch
python generate_submission.py --model-folder ${unixts} --resolution ${resolution} --batch-size ${batch_size}

