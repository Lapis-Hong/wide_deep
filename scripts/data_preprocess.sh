#!/usr/bin/env bash
# The scripts do the data preprocess using pyspark.

cd ../python/spark
# Run on a YARN cluster
# export HADOOP_CONF_DIR=XXX
# Configuration details see http://spark.apache.org/docs/latest/configuration.html
# cluster mode bug, use client
spark-submit \
  --master yarn \
  --deploy-mode client \  # deploy your driver on the worker nodes (cluster) or locally as an external client (client) (default: client)
  --name wide_deep \
  --driver-cores 8 \  # default 1, only in cluster mode.
  --executor-cores 8 \  # default 1
  --driver-memory 10g \  #default 1g
  --executor-memory 20g \
  --num-executors 100 \  # Number of executors to launch (Default: 2).
  --verbose \
  data_process.py


#  --conf spark.default.parallelism=1000  # 2~3 times num-executors * executor-cores
#  --conf spark.storage.memoryFraction=0.5
#  --conf spark.shuffle.memoryFraction=0.3 # default 0.2
#  --conf spark.cores.max=200 \