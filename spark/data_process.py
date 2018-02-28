#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/27
"""Using Spark to do Raw Data Preprocess
Raw data --> Processed raw data

Details:
    1. generate new continuous features (from category featrues)
        for each category, calculate target ratio under certain period as a new continuous feature
        here we use past 1-day, 7-days, 30-days as our periods
        for example:  
            label  sex          label  sex  new_column
              0     F    after    0     F     0.0
              0     M   ------>   0     M     0.33
              1     M             1     M     0.33
              0     M             0     M     0.33
        spark logic:
            1) filter target 2 columns and calculate all the ratio (1day, 7days, 30days) using RDD
            2) left join origin data and the new ratio column using DataFrame
            3) repeat 1) and 2) with different category column
            
    2. down-sampling
        ctr data is extremely unbalance, down sample for target 0
All configuration see conf/data_process.yaml
"""

# TODO: debug for run spark on yarn cluster and performance improvement
import os
from datetime import date, datetime, timedelta
import subprocess

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from read_conf import Config
from lib.util import timer

CONF = Config().read_data_process_conf()
SPARK_CONF = CONF['spark']
SCHEMA = Config().read_schema()


def gen_dates(start=None, days=1, fmt='%Y%m%d'):
    """generate date list before given date"""
    start = datetime.strptime(start, fmt)
    day = timedelta(days=1)
    return [(start-day*i).strftime(fmt) for i in range(days)]


def get_today():
    return date.today().strftime('%Y%m%d')  # default %Y-%m-%d fmt


@timer('Successfully process the raw data!')
def local_data_preprocess(inpath, outpath):
    """Use spark to process local data demo.
    Args:
        inpath: local data path
        outpath: local processed data path
    """
    feature_index_list = CONF['feature_index_list']
    conf = SparkConf().setAppName(SPARK_CONF['app_name']). \
        set('spark.executor.memory', SPARK_CONF['executor_memory']).set('spark.driver.memory', SPARK_CONF['driver_memory']). \
        set('spark.driver.cores', SPARK_CONF['driver_cores']).set('spark.cores.max', SPARK_CONF['cores_max'])
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.getOrCreate()
    rdd = sc.textFile(inpath).map(lambda x: x.strip().split('\t'))
    colnames = SCHEMA.values()
    df = ss.createDataFrame(rdd, colnames)
    # print(df.describe())

    for i in feature_index_list:
        # first filter target 2 columns, then groupByKey and calculate the mean of clk (ratio of click)
        # each rdd pair like [(u'150000', 0.0), (u'220000', 0.0), (u'130000', 0.00303951367781155)...]
        rdd2 = rdd.map(lambda x: (x[i-1], int(x[0]))).groupByKey().mapValues(lambda x: float(sum(x))/len(x))
        df2 = ss.createDataFrame(rdd2, ('k2', SCHEMA[i]+'_rate'))
        df = df.join(df2, df[SCHEMA[i]] == df2['k2'], how='left_outer')
        df = df.drop('k2')
    # down sampling
    df = df.sampleBy('clk', fractions={'0': 0.1, '1': 1}, seed=0)  # stratified sample without replacement

    # df.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(outpath)
    df.repartition(1).write.mode('overwrite').csv(outpath, sep='\t')
    sc.stop()
    ss.stop()


@timer('Successfully process the raw data!')
def hdfs_data_preprocess():
    """Use spark to process hdfs data and write result into hdfs
    """
    feature_index_list = CONF['feature_index_list']
    date = CONF['date']
    keep_prob = CONF['downsampling_keep_ratio']

    conf = SparkConf().setAppName(SPARK_CONF['app_name']). \
        set('spark.executor.memory', SPARK_CONF['executor_memory']).set('spark.driver.memory', SPARK_CONF['driver_memory']). \
        set('spark.driver.cores', SPARK_CONF['driver_cores']).set('spark.cores.max', SPARK_CONF['cores_max']).setMaster('yarn')
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.getOrCreate()

    if date is None:
        date = get_today()
    inpaths = [os.path.join(SPARK_CONF['input_hdfs_dir'], d) for d in gen_dates(date, 30)]
    outpath = os.path.join(SPARK_CONF['output_hdfs_dir'], date)

    # check path
    # hadoop output dir can not be exists, must be removed
    subprocess.call('hadoop fs -rm -r {0}'.format(outpath), shell=True)

    # 1-day period
    rdd = sc.textFile(inpaths[0]).map(lambda x: x.strip().split('\t'))
    colnames = SCHEMA.values()
    df = ss.createDataFrame(rdd, colnames)
    for i in feature_index_list:
        # first filter target 2 columns, then groupByKey and calculate the mean of clk (ratio of click)
        # each rdd pair like [(u'150000', 0.0), (u'220000', 0.0), (u'130000', 0.00303951367781155)...]
        rdd2 = rdd.map(lambda x: (x[i-1], int(x[0]))).groupByKey().mapValues(lambda x: float(sum(x))/len(x))
        df2 = ss.createDataFrame(rdd2, ('k2', SCHEMA[i]+'_rate_1'))
        df = df.join(df2, df[SCHEMA[i]] == df2['k2'], how='left_outer')
        df = df.drop('k2')
        print('1-day feature `{}` finished.'.format(SCHEMA[i]))

    # 7-days period
    for p in inpaths[1:]:
        rdd.union(sc.textFile(p).map(lambda x: x.strip().split('\t')))

    for i in feature_index_list:
        rdd2 = rdd.map(lambda x: (x[i - 1], int(x[0]))).groupByKey().mapValues(lambda x: float(sum(x)) / len(x))
        df2 = ss.createDataFrame(rdd2, ('k2', SCHEMA[i] + '_rate_7'))
        df = df.join(df2, df[SCHEMA[i]] == df2['k2'], how='left_outer')
        df = df.drop('k2')
        print('7-day feature `{}` finished.'.format(SCHEMA[i]))

    # 30-days period
    for p in inpaths[7:]:
        rdd.union(sc.textFile(p).map(lambda x: x.strip().split('\t')))

    for i in feature_index_list:
        rdd2 = rdd.map(lambda x: (x[i - 1], int(x[0]))).groupByKey().mapValues(lambda x: float(sum(x)) / len(x))
        df2 = ss.createDataFrame(rdd2, ('k2', SCHEMA[i] + '_rate_30'))
        df = df.join(df2, df[SCHEMA[i]] == df2['k2'], how='left_outer')
        df = df.drop('k2')
        print('30-day feature `{}` finished.'.format(SCHEMA[i]))

    # down sampling
    df = df.sampleBy('clk', fractions={'0': keep_prob, '1': 1}, seed=0)
    print('down sampling finished.')
    # data.saveAsTextFile(outpath, 'org.apache.hadoop.io.compress.GzipCodec')  # can not use snappy
    # data.repartition(1).saveAsTextFile(outfile)  # can be merged into 1 file, but for big data, it would be slow
    # df.write.mode('overwrite').save(outpath)
    # df.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(outpath)
    df.repartition(200).write.csv(outpath, sep='\t')
    sc.stop()
    ss.stop()


if __name__ == '__main__':
    # local path test
    inpath = 'file:////Users/lapis-hong/Documents/NetEase/wide_deep/data/train'
    outpath = 'file:////Users/lapis-hong/Documents/NetEase/wide_deep/data/spark'
    local_data_preprocess(inpath, outpath)
    #hdfs_data_preprocess()


