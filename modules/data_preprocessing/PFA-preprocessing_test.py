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

# # Geothermal and Machine Learning Sandbox

# ## this notebook is simply a demonstration for trying various preprocessing techniques and plotting the results

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import random

from scipy.stats import uniform

import datetime
import time

from pathlib import Path
import os, sys
# -

# %load_ext autoreload
# %autoreload 2

# ## import local function for preprocessing PFA data

# +
myPath = os.getcwd()
sys.path.append(myPath)

import PFA_preprocessing as pfa
# -

myPath

# ## load PFA data

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

# ## preprocess data

extraFeatures = ['GeochemistryTemperature2', 
                 'TufaOrTravertine_Distance',
                 'Silica_Distance',
                 'Hillshade-100m', 'DEM-30m']
# extraFeatures = ['HorizMagneticGradient2','DEM-30m']
# extraFeatures = None

# +
dfn, dfInfo, nullIndexes, scaler = pfa.preprocess_features_LocalNumerical(df_features, 
                                                                          resetLocal=None,
                                                                          transformFaultTraces='distance_edt',
                                                                          extraFeatures=extraFeatures, 
                                                                          prescaleFeatures=False)

dfc, dfInfoc, nullIndexesc, scalerc = pfa.preprocess_features_LocalCategorical(df_features.copy(), df_SSlookup,
                                                                               resetLocal=None,
                                                                               transformFaultTraces='distance_edt',
                                                                               extraFeatures=None,
                                                                               prescaleFeatures=True)

# transformDict = {}
transformDict = {'features': ['Local-StructuralSetting', 'QuaternaryFaultTraces'], 
                 'transforms': ['gaussianFilter', 'gaussianFilter'], 
                 'params': [5, 10]}

dfna, dfInfo, nullIndexes, scaler = pfa.preprocess_features_AllNumerical(df_features,
                                                                          transformFeatures=transformDict,
                                                                          extraFeatures=extraFeatures, 
                                                                          prescaleFeatures=False, withMean=False)
                                                                          # prescaleFeatures=False, withMean=True)



# -

dfna.columns

# +
fig, ax = plt.subplots(figsize=(6,4))

x = dfna['QuaternaryFaultTraces'].to_numpy()
x = np.reshape(x,(1000,-1))
plt.imshow(x)

# +
fig, ax = plt.subplots(figsize=(6,4))

x = dfn['Local-StructuralSetting'].to_numpy()
x = np.reshape(x,(1000,-1))
plt.imshow(x)
plt.title('raw')

# +
fig, ax = plt.subplots(figsize=(16,4))
plt.plot(x[200,:])

plt.title('row 200, sigma=5')
# plt.savefig('profileSigma5.png')
# -
# ### in preprocessing we could transform the target to account for a skewed distribution

x = dfna['HorizMagneticGradient2'].to_numpy()


# +
fig, ax = plt.subplots(figsize=(6,4))

plt.hist(x,50);

# +
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(standardize=True)

transformed = pt.fit_transform(x.reshape(-1,1))
y = transformed

print(pt.lambdas_)

# +
fig, ax = plt.subplots(figsize=(6,4))

plt.hist(y,50);
# -

# ### we can inverse transform (back) the data

yInverse = pt.inverse_transform(y.reshape(-1,1))

# +
fig, ax = plt.subplots(figsize=(6,4))

plt.hist(yInverse,50);

# +
fig, ax = plt.subplots(figsize=(12,4), nrows=1, ncols=2)

x = np.reshape(x,(1000,-1))
y = np.reshape(y,(1000,-1))

ax[0].imshow(x)
ax[1].imshow(y)
# -




