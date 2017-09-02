#!/bin/bash

unixts=$(date +%s)
resolution=256

python train.py --model-folder ${unixts} --resolution ${resolution}
python generate_submission.py --model-folder ${unixts} --resolution ${resolution}

