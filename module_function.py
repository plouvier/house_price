

import numpy as np
import pandas as pd
import os

## function to describe our data (missing value, mode, distribution, dimension...)
def describe(df):
    dim = df.shape
    nb_miss = df.apply(lambda x:x.isnull().sum())
    ttype_ = df.dtypes
    missing = (df.isnull().sum()/ df.shape[0]) * 100
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    kurto = df.kurt()
    mod=[]
    mod.append(1)
    for i in df.columns[1:]:
        mod.append(df[i].mode()[0])
    mod=pd.core.series.Series(mod)
    mod.index = counts.index
    description = pd.concat((nb_miss,ttype_,missing,counts,uniques, kurto, mod),axis=1,sort=False)
    description.columns=["nb_miss","type","missing","counts","uniques","kurto", "mode"]
    return description, dim

## function to check some value error or abberant value

def check_value_error(var, value_lim_min=None, value_lim_max=None, mod_liste=None):
    list_val_index = []
    count = 0
    if type(var[0]) in (int,float,np.int64,np.float64,np.int,np.float):
        for i in range(len(var)):
            if var[i]<value_lim_min or var[i]>value_lim_max:
                count +=1
                list_val_index.append(i)
    else:
        for i in range(len(var)):
            if var[i] not in (mod_liste):
                count +=1
                list_val_index.append(i)

    if count !=0:
        print("They are some error or aberrant values")

    return count,list_val_index


## function to have distribution of continue variable for each modality of a categorial variable
## usefull for make some test of independence 

def continu_mod(df,cont, cat,mod0,mod1,mod2=None,mod3=None,mod4=None,mod5=None,mod6=None,mod7=None,mod8=None,mod9=None,mod10=None):
    dic = {}
    for i in range(0,10):
        dic["cont_mod{0}".format(i)]= list(df[cont][list(np.where(df[cat]==eval("mod{0}".format(i)))[0])])

    print("this function work only with variables who have less 10 modality")
    return dic

