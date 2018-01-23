#!/usr/bin/env bash
set -e
train=/home/appops/data/train
test=/home/appops/data/test
nohup python train.py --train_data $train --test_data $test --batch_size 256 > log/wide_deep.log 2>&1 &
nohup python train.py --train_data $train --test_data $test --batch_size 256 --model_type wide > log/wide.log 2>&1 &
nohup python train.py --train_data $train --test_data $test --batch_size 256 --model_type deep > log/deep.log 2>&1 &
