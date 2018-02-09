#!/usr/bin/env bash
# kill ps process
ps -ef | grep python| grep train.py | awk {'print $2'} | xargs kill -9
