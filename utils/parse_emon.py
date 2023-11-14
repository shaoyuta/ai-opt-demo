import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def _get_emon_data(filename, fieldlist, sheetname=None):
    '''
    Pick field data from emon file, the fileds are from fieldlist.
    Return a dict
    # flist=["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz"]
    # r=_get_emon_data(emon_file, flist, "socket view")  => 
    {'metric_CPU operating frequency (in GHz)': 3.199832960295464, 'metric_uncore frequency GHz': 2.500640918227364}
    '''
    df = pd.read_excel(filename, sheet_name=sheetname, index_col=0, header=None)
    ret=dict()
    try:
        for f in fieldlist:
            ret[f]=df.loc[f][1]
    except:
        print(filename)
        pass
    return ret

def _parse_emon_filename(filename):
    '''
    Parse a filename of an emon file to a dict:
    # "amx_8_3.6Ghz.dat.xlsx"  =>
    # {"inst":"amx", "cores":8, "freq":3.6Ghz"}
    '''
    r=dict()
    l=filename.split('_')
    r["freq"]=l[-1][:-9]
    r["cores"]=l[-2]
    if l[0]=="simd":
        r["inst"]=l[0]+"_"+l[1]
    else:
        r["inst"]=l[0]
    return r

def parse_emon_fold(foldname, fieldlist, sheetname=None):
    '''
    Parse a set of emon file in one fold
    Return a list of element. The element is a dict 
    # flist=["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz",'metric_package power (watts)']
    # r=parse_emon_fold("/home/taosy/demo-emon/emon-spr/", flist, "socket view")  =>
    # [... {'freq': '3.0Ghz', 'cores': '20',  'inst': 'simd_sse',  'metric_CPU operating frequency (in GHz)': 3.000441233715261, ....}
    # ...]
    '''
    l=[]
    for fn in os.listdir(foldname):
        conf_info=_parse_emon_filename(fn)
        data=_get_emon_data(os.path.join(foldname,fn), fieldlist, sheetname)
        l.append(conf_info | data)
    return l

def save_result(r, fn):
    '''
    Save result as a picker file
    r=parse_emon_fold("/home/taosy/demo-emon/emon-spr/", flist, "socket view")
    # save_result(r, 'test.pickle')
    '''
    with open(fn, 'wb') as handle:
        pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_result(fn):
    '''
    Load result from a picker file
    # load_result('test.pickle')
    '''
    with open(fn, 'rb') as handle:
        r=pickle.load(handle)
    return r


def _filter_data(r, condition_list):
    '''
    Filter out the data which meet the requirements of condistion_list
    Return: 
        l: list of selected data (dict object)
        key_x: remaining condition
    # cl={"freq":'3.2Ghz', 'inst':"amx"}
    # l, key_x = _filter_data(r, cl) =>
    # key_x: "cores", l: list of data
    '''
    a={'freq', 'cores', 'inst'}
    key_x = a-set(condition_list.keys())
    l=[]
    for i in r:
        found=True
        for k in condition_list:
            found = found and (i[k]==condition_list[k])
        if found :
            l.append(i)
    return (l, next(iter(key_x)))


def _sort_by_X(l, X, Ys):
    '''
    Sort the object list by X, which is used for plot
    Return dX: series of column "X"
    Return dYs: serieses of specified by Ys
    # _sort_by_X(l, "cores", ["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz", 'metric_package power (watts)']) =>
    # dX : [ 4 8 12 ... 56 ]
    # dYs: [(2.896159689391512, 2.449375667625732, 233.6868085710844), .... ]
    '''
    def myFunc(e):
        v=e[X]          # FIXME: 
        try:
            return int(v)
        except:
            return v
    l.sort(key=myFunc)
    r=[]
    dX=[ i[X] for i in l ]
    for y in Ys:
        r.append([ i[y] for i in l ])
    dYs=list(zip(*r))
    
    return (dX,dYs)

def plot_emon_data(l, condition_list):
    '''
    plot_emon_data(l, {"freq":'3.2Ghz', 'inst':"amx"})
    '''
    lfilter, key_x=_filter_data(l, condition_list)
    (dX, dYs) = _sort_by_X(lfilter, key_x, ["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz", 'metric_package power (watts)'])
    d_cpu_freq = [i for i,j,k in dYs]
    d_uncore_freq = [j for i,j,k in dYs]
    d_power = [k for i,j,k in dYs]
    
    fig = plt.figure(tight_layout=False)
    fig.suptitle(str(condition_list))

    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(dX, d_cpu_freq, color='red', label="core freq")
    ax1.plot(dX, d_uncore_freq, color='blue', label="uncore freq")
    ax1.legend(shadow=False)
    ax1.set_ylabel('Actual Freq')
    ax1.set_ylim(1.4,4.0)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(dX, d_power,label="power")
    ax2.set_ylabel('Socket Power(Watt)')
    fig.set_figwidth(10)
    fig.set_figheight(3)
    ax2.set_ylim(180,370)
    ax2.legend(shadow=False)

    if key_x == "cores":
        ax1.set_xlabel('running cores')
        ax2.set_xlabel('running cores')
    elif key_x == "freq":
        ax1.set_xlabel('setting core freq')
        ax2.set_xlabel('setting core freq')
    elif key_x == "inst":
        ax1.set_xlabel('x86 instruction set')
        ax2.set_xlabel('x86 instruction set') 

    plt.show()
