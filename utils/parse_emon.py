import pandas as pd
import os
import pickle

def get_emon_data(filename, fieldlist, sheetname=None):
    '''
    # flist=["metric_CPU operating frequency (in GHz)", "metric_uncore frequency GHz"]
    # infolist={"freq":"2.4Ghz", "cores":4, "iset":"amx"}
    # r=get_emon_data(emon_file, flist, infolist, "socket view")
    {'freq': '2.4Ghz', 'cores': 4, 'iset': 'amx', 'metric_CPU operating frequency (in GHz)': 3.199832960295464, 'metric_uncore frequency GHz': 2.500640918227364}
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
    r=dict()
    l=filename.split('_')
    r["freq"]=l[-1][:-9]
    r["cores"]=l[-2]
    if l[0]=="simd":
        r["inst"]=l[0]+"_"+l[1]
    else:
        r["inst"]=l[0]
    return r

def _parse_emon_fold(foldname, fieldlist, sheetname=None):
    '''
    r=_parse_emon_fold("/home/taosy/demo-emon/emon-spr/", flist, "socket view")
    '''
    l=[]
    for fn in os.listdir(foldname):
        conf_info=_parse_emon_filename(fn)
        data=get_emon_data(os.path.join(foldname,fn), fieldlist, sheetname)
        l.append(conf_info | data)
    return l

def save_result(r, fn):
    '''
    r=_parse_emon_fold("/home/taosy/demo-emon/emon-spr/", flist, "socket view")
    save_result(r, 'test.pickle')
    '''
    with open(fn, 'wb') as handle:
        pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_result(fn):
    '''
    load_result('test.pickle')
    '''
    with open('fn', 'rb') as handle:
        r=pickle.load(handle)
    return r