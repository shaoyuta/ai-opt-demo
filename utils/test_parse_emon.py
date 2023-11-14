import pandas as pd
import os
import pickle
import parse_emon

flist=["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz", 'metric_package power (watts)']
emon_fold='/home/taosy/demo-emon/emon-spr/'

r=parse_emon._parse_emon_fold(emon_fold,flist, "socket view")
for i in r:
    print(i)

print(len(r))

parse_emon.save_result(r, "r.pickle")