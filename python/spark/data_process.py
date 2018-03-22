#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/27
"""Using Spark to do Raw Data Preprocess
Raw data --> Processed raw data
Data size is about 60G per day. 300M per part (200 part)
If use 0.01 down-sampling, use 2 part to avoid too much small partitions
Performance: 1 min one day one feature

Details:
    1. generate new continuous features from category featrues (optional)
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

# TODO: performance improvement
import os
import sys
import subprocess
from datetime import date, datetime, timedelta

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.utils.util import timer


def gen_dates(start, days=1, fmt='%Y%m%d'):
    """generate date list before given date."""
    start = datetime.strptime(start, fmt)
    day = timedelta(days=1)
    return [(start-day*i).strftime(fmt) for i in range(days)]


def list_dates(start, end, fmt='%Y%m%d'):
    """generate date list between start_date and end_date."""
    start = datetime.strptime(start, fmt)
    end = datetime.strptime(end, fmt)
    days = (end-start).days
    return [(start + timedelta(i)).strftime(fmt) for i in range(days+1)]


def get_today():
    return date.today().strftime('%Y%m%d')  # default %Y-%m-%d fmt


def exist_hdfs_path(path):
    """check hdfs path exists or not, return bool"""
    if subprocess.call('hadoop fs -test -e {}'.format(path), shell=True) == 0:
        return True
    else:
        return False


@timer('Successfully process the raw data!')
def hdfs_data_preprocess(inpath, outpath):
    """Use spark to process hdfs data and write result into hdfs
    """
    conf = SparkConf().setMaster("yarn")
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.getOrCreate()
    # generate 1-day, 7-days, 30-days rdd list
    rdd_list = [sc.textFile(inpath[0]), sc.textFile(','.join(inpath[:7])), sc.textFile(','.join(inpath))]

    data = rdd_list[0].map(lambda x: x.strip().split('\t'))
    if feature_index_list:  # if feature_index_l
        for rdd in rdd_list:
            for i in feature_index_list:
                # pair rdd (k, v) -> (category_f, clk)
                rdd = rdd.map(lambda x: x.strip().split('\t')).map(lambda x: (x[i-1], int(x[0])))
                # calculate the mean value of pair rdd by key
                # result like [(u'150000', 0.0), (u'220000', 0.0), (u'130000', 0.00303951367781155)...]
                # mehthod 1: reduceByKey, using reduceByKey much faster than groupByKey
                pair_rdd = rdd.mapValues(lambda v: (v, 1))\
                    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda v: float(v[0]) / v[1])
                # method 2:  groupByKey
                # pair_rdd = rdd.map(lambda x: (x[i-1], int(x[0]))).groupByKey().mapValues(
                #    lambda x: float(sum(x)) / len(x))
                # method 3: countByKey + reduceByKey
                # countsByKey = sc.broadcast(rdd.countByKey())  # SAMPLE OUTPUT of countsByKey.value
                # from operator import add
                # pair_rdd = rdd.reduceByKey(add).map(lambda x: (x[0], float(x[1]) / countsByKey.value[x[0]]))
                # method 4: aggregateByKey
                # pair_rdd = rdd.aggregateByKey((0, 0), lambda a, b: (a[0]+b, a[1]+1), lambda a,b: (a[0]+b[0], a[1]+b[1]))
                #    .mapValues(lambda v: float(v[0])/v[1])
                dic = pair_rdd.collectAsMap()
                # do not use append, modify inplace, return None
                # must persist(), or it will all use the last same dic
                # data = data.map(lambda x: x+[str(dic[x[i-1]])]).persist()
                b = sc.broadcast(dic)
                data = data.map(lambda x: x + [str(b.value[x[i - 1]])]).persist()
                # b.unpersist()
    # down sampling
    data = data.map(lambda x: (x[0], x)).sampleByKey('clk', fractions={'0': keep_prob, '1': 1}, seed=0)
    # rdd.saveAsTextFile(outpath, 'org.apache.hadoop.io.compress.GzipCodec')  # can not use snappy
    # merged into 1 file, but slow for big data coalesce(1)
    data.map(lambda x: "\t".join(x)).repartition(2).saveAsTextFile(outpath)
    sc.stop()
    ss.stop()


if __name__ == '__main__':
    CONF = Config().read_data_process_conf()
    SCHEMA = Config().read_schema()

    feature_index_list = CONF['category_feature_index_list']
    start_date = str(CONF['start_date'])
    end_date = str(CONF['end_date'])
    keep_prob = CONF['downsampling_keep_ratio']

    if start_date is None or end_date is None:
        date_list = [get_today()]
    else:
        date_list = list_dates(start_date, end_date)

    for date in date_list:
        print('Start processing date: {}'.format(date))
        inpath = [os.path.join(CONF['input_hdfs_dir'], d) for d in gen_dates(date, 30)]
        outpath = os.path.join(CONF['output_hdfs_dir'], date)
        # check path, hadoop output dir can not be exists, must be removed
        for p in inpath:
            if not exist_hdfs_path(p):
                raise IOError('Hdfs path: {} not exsits'.format(p))
        if exist_hdfs_path(outpath):
            subprocess.call('hadoop fs -rm -r {}'.format(outpath), shell=True)
            print('Remove hdfs path: {}'.format(outpath))

        hdfs_data_preprocess(inpath, outpath)


