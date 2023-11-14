# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Geothermal and Machine Learning Sandbox
# -

# # Prepare Nevada PFA Geothermal Resources Dataset

# +
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import datetime
import time

from pathlib import Path
import os, sys

from tqdm.notebook import trange, tqdm
# -
# ## import local function for preprocessing PFA data

# +
myPath = os.getcwd()
sys.path.append(myPath)

import PFA_preprocessing as pfa
# -

myPath

# ## load PFA data

# dataPath = myPath+'/../../../datasets/pandas_dataframes/'
dataPath = myPath+'/../../datasets/pandas_dataframes/'

dataPath

# +
filename = 'PFA_dataframe_grid_march2021_df.h5'

df_features = pd.read_hdf(dataPath+filename, 'df_features')

# +
filename = 'PFA_structuralSettingsLookup_march2021_df.h5'

df_SSlookup = pd.read_hdf(dataPath+filename, 'df_SSLookup')
# -

df_features.tail()

df_features.shape

df_SSlookup.head()

df_SSlookup.shape

df_features.columns

# ## preprocess data

extraFeatures = ['GravityDensity', 'MagneticDensity', 
                 'GeochemistryTemperature2',
                 'Silica_Distance', 'TufaOrTravertine_Distance',
                 'DEM-30m', 'FavorableStructuralSettings_Distance']
# extraFeatures = ['HorizMagneticGradient2','DEM-30m']
# extraFeatures = None

# +
# transformDict = {}

transformDict = {'features': ['Local-StructuralSetting',
                              'Local-QuaternaryFaultRecency',
                              'Local-QuaternaryFaultSlipDilation', 
                              'Local-QuaternaryFaultSlipRate',
                              'QuaternaryFaultTraces'], 
                 'transforms': ['gaussianFilter', 
                                'gaussianFilter', 
                                'gaussianFilter', 
                                'gaussianFilter', 
                                'distance_edt'], 
                 'params': [10, 10, 10, 10, 20]}

# transformDict = {'features': ['QuaternaryFaultTraces'], 
#                  'transforms': ['distance_edt'], 
#                  'params': [20]}

dfn, dfInfo, nullIndexes, scaler = pfa.preprocess_features_AllNumerical(df_features,
                                                                          transformFeatures=transformDict,
                                                                          extraFeatures=extraFeatures, 
                                                                          prescaleFeatures=True, withMean=True)
#                                                                           prescaleFeatures=True, withMean=False)

# dfn, dfInfo, nullIndexes, scaler = pfa.preprocess_features_LocalNumerical(df_features, 
#                                                                           resetLocal=None,
#                                                                           transformFaultTraces='distance_edt',
#                                                                           extraFeatures=extraFeatures, 
#                                                                           prescaleFeatures=False)

# dfc, dfInfoc, nullIndexesc, scalerc = pfa.preprocess_features_LocalCategorical(df_features.copy(), df_SSlookup,
#                                                                                resetLocal=None,
#                                                                                transformFaultTraces='distance_edt',
#                                                                                extraFeatures=None,
#                                                                                prescaleFeatures=True)
# -

nFeatures = len(dfn.columns)
print (nFeatures)

# +
# dfn = dfn.drop(dfn.columns[0:4], axis=1)

# +
# categorical structural settings

# dfc, dfInfo, nullIndexes, scaler = pfa.preprocess_features_LocalCategorical(df_features, df_SSlookup,
#                                                                              resetLocal=None,
#                                                                              extraFeatures=extraFeatures)
# dfc, dfInfo, nullIndexes, scaler = pfa.preprocess_features_LocalCategorical(df_features, df_SSlookup,
#                                                                              df_features, 
#                                                                              resetLocal='random')

# +
# print (len(scaler))
# print ('')
# print (scaler[0].scale_)
# print (scaler[0].mean_)
# print (scaler[0].var_)
# -
dfn.columns

# ## select benchmark sites based on trainCode distance

# ### set random number seed

seed = 10

# +
np.random.seed(seed)

# X_pfa, y_pfa, XyInfo = pfa.makeBenchmarks(dfn, dfInfo, nullIndexes, 
#                                           trainCode=2, randomize=True, balance=True)

X_pfa, y_pfa, XyInfo = pfa.makeBenchmarks(dfn, dfInfo, nullIndexes, 
                                          trainCode=2, randomize=True, balance=False)



# +
# option to save all data info so that we can extract row,col locations of benchmark sites for plotting

# X_pfa, y_pfa, XyInfo = pfa.makeBenchmarks(dfn, dfInfo, nullIndexes, 
#                                           trainCode=1, randomize=False, balance=False)

# # hf5File = 'benchmark_sites_february2021_tc2_df.h5'
# hf5File = 'benchmark_sites_february2021_tc1_df.h5'

# XyInfo.to_hdf(hf5File, 'XyInfo', format='table', mode='a')
# -

print( X_pfa.shape, y_pfa.shape)

X_pfa.head()

y_pfa.head()

XyInfo

dfInfo

# ## write dataframes to hdf file archive

print( X_pfa.shape, y_pfa.shape, XyInfo.shape)

# +
filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

# place file in folder one level up
hf5File = dataPath+'/../'+filename
# -

hf5File

# +
####################################################
# add benchmark data
X_pfa.to_hdf(hf5File, 'X', format='table', mode='a')
y_pfa.to_hdf(hf5File, 'y', format='table', mode='a')
XyInfo.to_hdf(hf5File, 'XyInfo', format='table', mode='a')

####################################################
# add whole study area dataframes and info for inference
dfn.to_hdf(hf5File, 'dfn', format='table', mode='a')
dfInfo.to_hdf(hf5File, 'dfInfo', format='table', mode='a')

dfNullIndexes = pd.Series(nullIndexes)
dfScaler = pd.DataFrame(np.array([scaler[0].scale_, scaler[0].mean_, scaler[0].var_]))

dfNullIndexes.to_hdf(hf5File, 'nullIndexes', format='table', mode='a')
dfScaler.to_hdf(hf5File, 'scaler', format='table', mode='a')                        
                        
# -

# ## check contents of archive

# +
import h5py
f = h5py.File(hf5File, 'r')

def keys(f):
    return [key for key in f.keys()]

key_list = keys(f)
print(key_list)

f.close()
# -

X_tst = pd.read_hdf(hf5File, key='X')
y_tst = pd.read_hdf(hf5File, key='y')
XyInfo_tst = pd.read_hdf(hf5File, key='XyInfo')

xx_tst = pd.read_hdf(hf5File, key='nullIndexes')

xx_tst

print( X_tst.shape, y_tst.shape, XyInfo_tst.shape)

X_tst.head()

y_tst.head()

XyInfo_tst.head()




hf5File




