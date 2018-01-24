#!/bin/bash
hosts=(dinghongquan@10.172.110.162 dinghongquan@10.120.180.212 dinghongquan@10.120.180.213
dinghongquan@10.120.180.214 dinghongquan@10.120.180.215)
dir=/home/dinghongquan/wide_deep

for host in ${hosts[@]}
do
		rsync -rvz -e'ssh -p 1046' --exclude 'model' --exclude 'log' --exclude 'conf' ./* $host:$dir
done
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.172.110.162:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.212:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.213:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.214:/home/dinghongquan/wide_deep
#rsync -rvz -e'ssh -p 1046' --exclude 'model' ./* dinghongquan@10.120.180.215:/home/dinghongquan/wide_deep
