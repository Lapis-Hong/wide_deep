#!/bin/bash
# The script load processed raw data from hdfs path to appops path.
# Usage: 
#		1. bash load_process_data.sh  # default to load yesterday data
#       2. bash load_process_data.sh  20180110  # one param $1 to load certain date data
#       3. bash load_process_data.sh  20180110 20180120  # two params $1 $2 to load data from $1 to $2

set -e

dt=`date -d "yesterday" +%Y%m%d`
end_dt=$dt
if [ $# -eq 1 ]; then
    dt=$1
    end_dt=$dt
elif [ $# -eq 2 ]; then
    dt=$1
    end_dt=$2
fi

from_hdfs_dir=/user/algo/user/dinghongquan/raw_data_downsample_0.01
to_local_dir=/home/appops/data/train

if [ ! -d "$to_local_dir" ]; then  
  sudo -iu appops mkdir -p "$to_local_dir" && echo "Already make local dir: $to_local_dir"
fi
	
function load_data() {
		dt=$1
		input_path=${from_hdfs_dir}/${dt}
        output_path=${to_local_dir}/${dt}
		echo "Start loading data from hdfs: ${input_path}."
        #sudo -iu appops hadoop fs -text ${input_path}/part* > ${output_path}
        sudo -iu appops hadoop fs -getmerge ${input_path}/part* ${output_path}
        sudo -iu appops chmod 666 ${output_path}
		echo "Finish loading data to local: ${output_path}."
}
if [ $dt -ne $end_dt ]; then
    cur_dt=$dt
    while [ $cur_dt -le $end_dt ]
    do
	    load_data $cur_dt
		cur_dt=`date -d "$cur_dt 1days" +%Y%m%d`
    done
else
		load_data $dt
fi

echo "Done! see data in ${to_local_dir}." 
cd ${to_local_dir}

