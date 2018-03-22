#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/1
"""This script is for spark local testing, process local data 
Using spark local[*] mode, run Spark locally with as many worker threads as logical cores on your machine.
see http://spark.apache.org/docs/latest/submitting-applications.html#master-urls

Usage $ python data_process_local_test.py $inpath $outpath
"""
import os
import shutil
import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.utils.util import timer


@timer('Successfully process the raw data!')
def local_data_preprocess(inpath, outpath):
    """Use spark to process local data demo.
    Args:
        inpath: local data path
        outpath: local processed data path
    """
    rdd = sc.textFile('file://{}'.format(inpath)).map(lambda x: x.strip().split('\t'))
    colnames = SCHEMA.values()
    df = ss.createDataFrame(rdd, colnames)

    for i in feature_index_list:
        # first filter target 2 columns, then groupByKey and calculate the mean of clk (ratio of click)
        # each rdd pair like [(u'150000', 0.0), (u'220000', 0.0), (u'130000', 0.00303951367781155)...]
        rdd2 = rdd.map(lambda x: (x[i-1], int(x[0]))).groupByKey().mapValues(lambda x: float(sum(x))/len(x))
        df2 = ss.createDataFrame(rdd2, ('k2', SCHEMA[i] + '_rate_1'))
        df = df.join(df2, df[SCHEMA[i]] == df2['k2'], how='left_outer')
        df = df.drop('k2')
        print('1-day feature `{}` finished.'.format(SCHEMA[i]))

    # down sampling
    df = df.sampleBy('clk', fractions={'0': 0.1, '1': 1}, seed=0)  # stratified sample without replacement
    df.repartition(1).write.mode('overwrite').csv(outpath, sep='\t')
    sc.stop()
    ss.stop()


@timer('Successfully process the raw data!')
def local_data_preprocess2(inpath, outpath):
    """Using RDD API, much more effcient way"""
    rdd = sc.textFile(inpath).map(lambda x: x.strip().split('\t'))

    data = rdd
    if feature_index_list:
        for i in feature_index_list:
            # method 1
            pair_rdd = rdd.map(lambda x: (x[i-1], int(x[0]))).mapValues(lambda v: (v, 1))\
                .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda v: v[0] / v[1])
            # method 2
            # pair_rdd = rdd.map(lambda x: (x[i-1], int(x[0]))).groupByKey().mapValues(
            #    lambda x: float(sum(x)) / len(x))
            # method 3
            # countsByKey = sc.broadcast(rdd.countByKey())  # SAMPLE OUTPUT of countsByKey.value
            # from operator import add
            # pair_rdd = rdd.reduceByKey(add).map(lambda x: (x[0], x[1] / countsByKey.value[x[0]]))
            # method 4
            # pair_rdd = rdd.aggregateByKey((0, 0), lambda a, b: (a[0]+b, a[1]+1), lambda a,b: (a[0]+b[0], a[1]+b[1]))
            #    .mapValues(lambda v: v[0]/v[1])
            dic = pair_rdd.collectAsMap()
            # do not use append, modify inplace, return None
            # must persist(), or it will all use the last same dic
            data = data.map(lambda x: x + [str(dic[x[i-1]])]).persist()
            # b = sc.broadcast(dic)
            # data = data.map(lambda x: x + [str(b.value[x[i - 1]])]).persist()
            # b.upersist()
            print('feature `{}` finished.'.format(SCHEMA[i-1]))
    # down sampling, first map to pair rdd
    data = data.map(lambda x: (x[0], x)).sampleByKey('clk', fractions={'0': keep_prob, '1': 1}, seed=0).values()
    print('down sampling finished.')
    print(data.first())
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    data.map(lambda x: "\t".join(x)).saveAsTextFile(outpath)
    sc.stop()
    ss.stop()


if __name__ == '__main__':
    CONF = Config().read_data_process_conf()
    SCHEMA = Config().read_schema()
    feature_index_list = CONF['category_feature_index_list']
    keep_prob = CONF['downsampling_keep_ratio']
    conf = SparkConf().setAppName('wide_deep'). \
        set('spark.executor.memory', '10g').set('spark.driver.memory', '10g').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.getOrCreate()
    inpath = '/Users/lapis-hong/Documents/NetEase/wide_deep/data/train'
    outpath = '/Users/lapis-hong/Documents/NetEase/wide_deep/data/spark'
    # if len(sys.argv) < 3:
    #     exit('Missing arguments: \nUsage: $ python data_process_local_test.py $inpath $outpath')
    if len(sys.argv) == 3:
        inpath = sys.argv[1]
        outpath = sys.argv[2]
    local_data_preprocess2(inpath, outpath)
