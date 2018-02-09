#!/bin/bash
# The scripts to synchronize wide and deep project between servers.
# Usage:
#		1. bash rsync.sh

hosts=(dinghongquan@10.172.110.162 dinghongquan@10.120.180.212 dinghongquan@10.120.180.213
dinghongquan@10.120.180.214 dinghongquan@10.120.180.215)


local_dir=`pwd`
remote_dir=/home/dinghongquan

#local_dir=./*
#remote_dir=/home/dinghongquan/wide_deep

for host in ${hosts[@]}
do
	rsync -rvz -e'ssh -p 1046' --exclude ".git" --exclude "model" --exclude "log" --exclude "conf/train.yaml" $local_dir $host:$remote_dir
done
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.172.110.162:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.212:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.213:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.214:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.215:/home/dinghongquan/wide_deep
