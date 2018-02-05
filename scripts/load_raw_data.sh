#!/bin/bash
# The script load raw data from hdfs path to appops path.
# Usage:
#		1. bash load_raw_data.sh  # default to load yesterday data
#       2. bash load_raw_data.sh  20180110  # one param $1 to load certain date data


set -e
input_hdfs_dir=/user/algo/algo_fea/v1/feature_joiner
output_local_dir=/home/appops/data/test

if [ $# -eq 0 ];then
    dt=`date -d "yesterday" +%Y%m%d`
elif [ $# -eq 1 ];then
    dt=$1
fi

echo "Input hdfs path: $input_hdfs_dir/$dt."
echo "Output local path: $output_local_dir/$dt."

if [ ! -d $output_local_dir/$dt ]; then
		sudo -iu appops mkdir -p $output_local_dir/$dt && echo "Make local path $output_local_dir/$dt"
fi

sudo -iu appops hadoop fs -get $input_hdfs_dir/$dt/part* $output_local_dir/$dt
sudo -iu appops chmod 666 $output_local_dir/$dt/*
echo "$?"
