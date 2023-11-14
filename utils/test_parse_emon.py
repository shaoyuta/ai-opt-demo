import pandas as pd
import os
import pickle
import parse_emon
import argparse

flist=["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz", 'metric_package power (watts)']
emon_fold='/home/taosy/demo-emon/emon-spr/'


parser = argparse.ArgumentParser('Emon data parser tester', add_help=False)
parser.add_argument("--test", default=1, type=int, help="test plan")

args = parser.parse_args()


def test_fun1():
    r=parse_emon._parse_emon_fold(emon_fold,flist, "socket view")
    for i in r:
        print(i)
    print(len(r))
    parse_emon.save_result(r, "r.pickle")

def test_fun2():
    r=parse_emon.load_result("r.pickle")
    print(len(r))

def test_fun3():
    r=parse_emon.load_result("r.pickle")
    parse_emon.plot_emon_data(r, {"freq":'3.2Ghz', 'inst':"amx"}, ["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz", 'metric_package power (watts)'])


if   args.test==1:
    test_fun1()
elif args.test==2:
    test_fun2()