import collections


DEFAULT_FILTER= {
    int:["*"],
    str:["*"],
    bool:["*"],
    set:["*"],
    list:["__jit_unused_properties__"],
    dict:["_methods"],
    collections.OrderedDict:["*"],
}


class kinfo():
    def __init__(self, k, vtype):
        self.k=k
        self.vtype=vtype
    def __str__(self):
        return "%30s:%30s"%(self.k, self.vtype)
    def __repr__(self):
        return "%30s:%30s"%(self.k, self.vtype)
    
def _canbe_filter_out(ki, filter):
    if ki.vtype not in filter.keys():
        return False
    else:
        fl = filter[ki.vtype]
        for i in fl:
            if i == "*":
                return True
            elif i==ki.k :
                return True
        return False


class comp_ret():
    def __init__(self, obj1, obj2):
        self.obj1=obj1
        self.obj2=obj2
        self.k_removed=set(dir(obj1)) - set(dir(obj2))
        self.k_added=set(dir(obj2)) - set(dir(obj1))
        self.k_both=set(dir(obj1)) & set(dir(obj2))
        self.added=_split_obj_key(obj2, self.k_added)
        self.removed=_split_obj_key(obj1, self.k_removed)
        self.both=_split_objs_key(self.obj1, self.obj2, self.k_both)
    
    def filter_out_added(self, filter=DEFAULT_FILTER, whitelist=None):
        self.added_filter=dict()
        for keys in ["var", "callable"]:
            self.added_filter[keys]=list()
            for ki in self.added[keys]:
                if not _canbe_filter_out(ki, filter):
                    self.added_filter[keys].append(ki)

    def filter_out_removed(self, filter=DEFAULT_FILTER, whitelist=None):
        self.removed_filter=dict()
        for keys in ["var", "callable"]:
            self.removed_filter[keys]=list()
            for ki in self.removed[keys]:
                if not _canbe_filter_out(ki, filter):
                    self.removed_filter[keys].append(ki)


    def filter_out_both(self, filter=DEFAULT_FILTER, whitelist=None):
        self.both_filter=dict()
        for keys in ["type_diff", "all-same", "var-diff", "callable"]:
            self.both_filter[keys]=list()
            for ki in self.both[keys]:
                if not _canbe_filter_out(ki, filter):
                    self.both_filter[keys].append(ki)

    def filter_out(self, filter=DEFAULT_FILTER, whitelist=None):
        self.filter_out_added(filter, whitelist)
        self.filter_out_removed(filter, whitelist)
        self.filter_out_both(filter, whitelist)

    

class _diff_ret(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        print("++repr++")
        return dictrepr

    def __str__(self):
        print("++str++")
        dictrepr = dict.__str__(self)
        return dictrepr
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


def _split_obj_key(obj,kl):
    ret=dict()
    ret["var"]=list()
    ret["callable"]=list()
    for k in kl:
        if callable(getattr(obj,k)):
            ret["callable"].append(kinfo(k,type(getattr(obj,k))))
        else:
            ret["var"].append(kinfo(k,type(getattr(obj,k))))
    return ret

def _split_objs_key(obj1, obj2, kl):
    ret=dict()
    ret["type_diff"]=[]
    ret['all-same']=[]
    ret['var-diff']=[]
    ret['callable']=[]

    for k in kl:
        if type(getattr(obj1,k)) != type(getattr(obj2,k)):  #case1: type is diff
            ret["type_diff"].append(kinfo(k,type(getattr(obj1,k))))
        else:
            if not callable(getattr(obj1,k)):  # case2: not callable object
                try: 
                    if getattr(obj1,k) is getattr(obj2,k):  # 2.1 check the type
                        ret["all-same"].append(kinfo(k,type(getattr(obj1,k))))
                    else:                                   # 2.2 check the value
                        ret['var-diff'].append(kinfo(k,type(getattr(obj1,k))))
                except:
                    print("except:===", k)
            else:
                try:
                    if getattr(obj1,k).__code__ is getattr(obj2,k).__code__:
                        ret["all-same"].append(kinfo(k,type(getattr(obj1,k))))
                    else:
                        ret['callable'].append(kinfo(k,type(getattr(obj1,k))))
                except:
                    ret['callable'].append(kinfo(k,type(getattr(obj1,k))))
    return ret
