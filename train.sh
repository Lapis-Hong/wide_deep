#!/usr/bin/env bash
# The script train the model.
# Usage:
#		1. bash train.sh  # default to train all 3 type model
#       2. bash train.sh  $model_name  # explict model name to train that type model.

set -e

if [ $# -eq 0 ]; then
    nohup python train.py --model_type wide_deep > log/wide_deep.log 2>&1 &
    nohup python train.py --model_type wide > log/wide.log 2>&1 &
    nohup python train.py --model_type deep > log/deep.log 2>&1 &
elif [ $# -eq 1 ]; then
    nohup python train.py --model_type $1 > log/$1 2>&1 &
fi


