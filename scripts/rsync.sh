#!/bin/bash
# The scripts to synchronize wide and deep project between servers.
# Usage:
#		1. bash rsync.sh

hosts=(dinghongquan@10.172.110.162 dinghongquan@10.120.180.212 dinghongquan@10.120.180.213
dinghongquan@10.120.180.214 dinghongquan@10.120.180.215)

cur_dir=$(cd `dirname $0`; pwd)
local_dir=`dirname $cur_dir`
remote_dir=/home/dinghongquan/wide_deep

echo local dir: $local_dir
echo remote dir: $remote_dir

for host in ${hosts[@]}
do
	rsync -rvz -e'ssh -p 1046' --exclude "$local_dir/.git" --exclude "$local_dir/model" --exclude "$local_dir/log" --exclude "$local_dir/conf/train.yaml" $local_dir $host:$remote_dir
done
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.172.110.162:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.212:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.213:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.214:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.215:/home/dinghongquan/wide_deep
