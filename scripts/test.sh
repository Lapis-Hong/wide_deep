#!/usr/bin/env bash
# The script train the model.
# Usage:
#		1. bash test.sh  # default to train all 3 type model
#       2. bash test.sh  $model_name  # explict model name to train that type model.

set -e

cur_dir=$(cd `dirname $0`; pwd)
log_dir=`dirname ${cur_dir}`/log

cd `dirname ${cur_dir}`/python

if [ $# -eq 0 ]; then
    nohup python test.py --model_type wide_deep > ${log_dir}/wide_deep_test.log 2>&1 &
    nohup python test.py --model_type wide > ${log_dir}/wide_test.log 2>&1 &
    nohup python test.py --model_type deep > ${log_dir}/deep_test.log 2>&1 &
elif [ $# -eq 1 ]; then
    nohup python test.py --model_type $1 > ${log_dir}/$1_test.log 2>&1 &
fi
