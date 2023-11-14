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

# # Skeleton of an Artificial Neural Network (ANN) in PyTorch applied to Nevada PFA Geothermal Resources Dataset

# +
import numpy as np
import matplotlib.pyplot as plt

import math

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import kurtosis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load packages to create dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torch import Tensor

# from torchvision import models
from torchsummary import summary

import datetime
import time

from pathlib import Path
import sys

import h5py

from tqdm.notebook import trange, tqdm
# -

# ## BEGIN data preprocessing

# ## load preprocessed data and trained model

# +
# myHome = str(Path.home())
# -

pwd

# +
path = '../../datasets/'
modelPath = './'

filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

# modelFilename = 'ANN_model_trial_dropout-0.2_weight_decay-0.1.torch'
modelFilename = 'ANN_model_trial_dropout-0.2_weight_decay-0.01.torch'

hf5File = path+filename
modelFile = modelPath+modelFilename


# +
f = h5py.File(hf5File, 'r')

def keys(f):
    return [key for key in f.keys()]

key_list = keys(f)
print(key_list)

f.close()

# +
dfn = pd.read_hdf(hf5File, key='dfn')
nullIndexes = pd.read_hdf(hf5File, key='nullIndexes')

# dfXAll = pd.read_hdf(hf5File, key='X')
# dfyAll = pd.read_hdf(hf5File, key='y')
# XyInfo = pd.read_hdf(hf5File, key='XyInfo')
# -

print( dfn.shape, nullIndexes.shape)

dfn.head()

nullIndexes.head()

columns=dfn.columns.to_list()

columns

# ## select feature set

# +
featureSets = [

# MASTER SET 1
################################# 0
['QuaternaryFaultTraces',
 'HorizGravityGradient2',
 'HorizMagneticGradient2',
 'GeodeticStrainRate',
 'QuaternarySlipRate',
 'FaultRecency',
 'FaultSlipDilationTendency2',
 'Earthquakes',
 'Heatflow',
 'DEM-30m',
],

]

print (len(featureSets))

# +
feature_set = 0

columns = featureSets[feature_set]

columns
# -

# ## END data preprocessing

# ## predict over study area

Xall = dfn.copy()

Xall.head()

# ## select only those features used for training

Xall = Xall[columns]

Xall.head()

len(Xall)

Xall = Xall.copy().to_numpy()

Xall.shape

# # Build model

# +
# setting DEVICE on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
print('Using DEVICE:', DEVICE)
print()

#Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# -



# ## FCNN in pytorch

# +
class FCNN(nn.Module):
    def __init__(self, nFeatures, nHidden, nLabels):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nLabels = nLabels
        
        self.fc1   = nn.Linear(self.nFeatures, self.nHidden)
        self.fc2   = nn.Linear(self.nHidden, self.nHidden)
        self.fc3   = nn.Linear(self.nHidden, nLabels)
        
#         self.Relu = nn.ReLU()
        self.LeakyRelu = nn.LeakyReLU(0.1)
        self.BatchNorm = nn.BatchNorm1d(self.nHidden)
        self.dropout = nn.Dropout(p=0.2)
        # self.dropout = nn.Dropout(p=0.0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = x.view(-1, self.nFeatures)
        out = self.LeakyRelu(self.fc1(out))
#         out = self.Relu(self.fc1(out))
        out = self.dropout(out)
        out = self.BatchNorm(out)
        out = self.LeakyRelu(self.fc2(out))
#         out = self.Relu(self.fc2(out))
        out = self.dropout(out)
    
        ######################################
        out = self.fc3(out) # these are logits
        
        ######################################
        out = self.LogSoftmax(out) 
        # output of network is LogSoftmax 
        # this makes probabilty=torch.exp(out)
        # proper loss criterion is used accordingly
        
        return out
# -



# ## load trained model

# +
net = torch.load(modelFile)

net.eval()

summary(net, Xall.shape, col_names=("input_size", "output_size", "num_params"), verbose=2, depth=2);
# -

# # Inference

# ## predict whole study area

# ### two approaches possible: 
#  - (1) sample entire study area many times to derive mean, stddev, and confidence - fast
#  - (2) sample entire study area many times to compute cdf, then derive mode and quantiles - slow

# +
# this saves memory on gpu, but is slow and doesn't work well with batchnorm

# eval_loader = DataLoader(Tensor(Xall).to(DEVICE), shuffle=False, 
#                          batch_size=8192, drop_last=False)

# net.eval()
# net.to(DEVICE)

# pm = []
# ps = []
# for features in tqdm(eval_loader):
#     with torch.no_grad():
#         p = net.mfvi_forward(features, sample_nbr=1024)
#         m = p[0].squeeze() # means
#         s = p[1].squeeze() # stddevs

#         pm.append(m)
#         ps.append(s)

# m = torch.cat(pm, dim=0)
# s = torch.cat(ps, dim=0)

# +
# net.eval()

# net.to('cuda')l
# features = Tensor(Xall).to('cuda')

# t0 = time.time()
# with torch.no_grad():
#     p = net.mfvi_forward(features, stat='mean', q=0.1, sample_nbr=64)
#     m = p.squeeze() # means
# #     p2 = net.mfvi_forward(features, stat='stddev', q=0.1, sample_nbr=64)
# #     s = p2.squeeze() # means
# print ('Inference time = ', time.time()-t0)
# -

# ## this ANN version does logsoftmax internally and outputs log(probabilities)

# +
# inference in batches
# this saves memory on gpu, but is slow 
# and has irregularities due to random number seeds among batches

eval_loader = DataLoader(Tensor(Xall).to(DEVICE), shuffle=False, 
                         batch_size=4096, drop_last=False)

###########################################
t0 = time.time()
###########################################

net.eval()
net.to(DEVICE)

pProbAll = []
for features in tqdm(eval_loader):
    with torch.no_grad():
        
        p = net.forward(features)
        
        p = p.squeeze()
        
        # convert to probability
        p = torch.exp(p)
        pProbAll.append(p)
        
pProbAll = torch.cat(pProbAll, dim=0)

###########################################
print ('Inference time = ', time.time()-t0)
###########################################
# -

pProbAll.shape

pProbAll = pProbAll.cpu().detach().numpy()
pProbAll.shape

# ## plot a histogram

# +
plt.rc('figure', figsize=(16,4))

plt.subplot(121)
plt.hist(pProbAll[:,0],50);
plt.grid(True)

plt.title('Predictions for study area',fontsize=20)
plt.xlabel('Probability of (-)',fontsize=18)

plt.subplot(122)
plt.hist(pProbAll[:,1],50);
plt.grid(True)

plt.title('Predictions for study area',fontsize=20)
plt.xlabel('Probability of (+)',fontsize=18)

plt.tight_layout()

# plt.savefig('prob_predict_study_area_fixedComplexityWeights.png')

# +
# choose which site type to look at

siteType = 1

if siteType == 0:
    # negative
    classLabel = '(-)'
    pProb = pProbAll[:,0].copy()
    
elif siteType == 1:
    # positive
    classLabel = '(+)'
    pProb = pProbAll[:,1].copy()


# -

# ## plot maps

# +
# mask nulls for image

pProb[nullIndexes] = np.nan

# +
img0 = np.reshape(pProb,(1000,-1))

img0.shape

# +
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(12,6))

im = plt.imshow(img0, origin='lower', cmap='coolwarm', 
                vmin=0, vmax=1.0,
                # vmin=0, vmax=0.25,
#                 interpolation='gaussian',
#                 norm=colors.Normalize(vmin=0.0, vmax=1.0)
#                 norm=colors.SymLogNorm(linthresh=0.3, linscale=0.3, vmin=-1.0, vmax=1.0, base=10))
               )

# plt.title('"new" Fairway - categorical localK', fontsize=18)
# plt.title('Bayesian Fairway, quantile = '+str(quantile), fontsize=18)
# plt.title('Bayesian Fairway, mean', fontsize=18)
# plt.title('Bayesian Fairway, stddev', fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

cb = plt.colorbar(im, cax=cax)
cb.set_label('probability of '+classLabel,fontsize=16)

plt.tight_layout()

# plt.savefig('nFairway_'+outFileRoot+'_'+classLabel+'_quantile_'+str(quantile)+'_avg_8192'+'_600dpi'+'.png', dpi=600)
# plt.savefig('nFairway_'+outFileRoot+'_'+classLabel+'_mean'+'_avg_8192'+'_600dpi'+'.png', dpi=600)
# plt.savefig('nFairway_'+outFileRoot+'_'+classLabel+'_stddev'+'_avg_8192'+'_600dpi'+'.png', dpi=600)

# plt.savefig('figures/statsMaps/nFairway_'+outFileRoot+'_'+classLabel+'_percentile_'+str(quantile)+'.png')
# plt.savefig('nFairway_numericalLocalK_finetune.png')
# plt.savefig('nFairway_categoricalLocalK_finetune.png')
# plt.savefig('nFairway_smoothedLocalK.png')
# -
# ## plot benchmark sites onto map

XyInfo = pd.read_hdf(hf5File, key='XyInfo')
# XyInfo

benchmarks = XyInfo[['row', 'column', 'TrainCodePos']].copy()
benchmarks.rename({'TrainCodePos': 'label'}, axis='columns', inplace=True)

# +
benchmarks.loc[benchmarks.label <= 2, 'label'] = 1
benchmarks.loc[benchmarks.label > 2, 'label'] = 0

benchmarks

# +
benchmarks = benchmarks.astype(int)

benchmarks = benchmarks.to_numpy()

# -

row = benchmarks[:,0]
col = benchmarks[:,1]
label = benchmarks[:,2]

# +
fig, ax = plt.subplots(figsize=(12,6))

im = plt.imshow(img0, origin='lower', cmap='coolwarm', 
                vmin=0.0, vmax=1.0,
#                 interpolation='gaussian',
#                 norm=colors.Normalize(vmin=0.0, vmax=1.0)
#                 norm=colors.SymLogNorm(linthresh=0.3, linscale=0.3, vmin=-1.0, vmax=1.0, base=10))
               )

plt.scatter(col, row, c=label, s=10, cmap='gray')

# plt.title('"new" Fairway - categorical localK', fontsize=18)
# plt.title('Bayesian Fairway, percentile = '+str(quantile), fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

cb = plt.colorbar(im, cax=cax)
cb.set_label('posterior probability of '+classLabel,fontsize=16)

plt.tight_layout()

# plt.savefig('figures/statsMaps/nFairway_'+'benchmarks_'+outFileRoot+'_'+classLabel+'_percentile_'+str(quantile)+'.png')
# plt.savefig('figures/nFairway_with_benchmarks_featureSet2.png', dpi=300)
# -

xxx

# ## save maps as geotiffs


import osgeo.gdal as gdal
import osgeo.osr as osr

transform = [286202.6902, 250.0, 0.0, 4508588.7659, 0.0, -250.0]

img0 = np.flipud(img0)
# img1 = np.flipud(imgm)
# img2 = np.flipud(imgmax)

height = img0.shape[0]
width = img0.shape[1]
print (width, height)

# +
# open a new file
# geotiffFile = 'figures/statsMaps/nFairway_'+outFileRoot+'_'+classLabel+'_percentile_'+str(quantile)+'.tif'
# geotiffFile = 'nFairway_'+outFileRoot+'_'+classLabel+'_quantile_'+str(quantile)+'_avg_8192'+'.tif'

# geotiffFile = 'nFairway_'+outFileRoot+'_'+classLabel+'_mean'+'_avg_8192'+'.tif'
geotiffFile = 'ANN_'+classLabel+'.tif'

driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(geotiffFile, width, height, 1, gdal.GDT_Float32)

# use PFA's geo-transform found above
outRaster.SetGeoTransform(transform)

# assign color band
outband = outRaster.GetRasterBand(1)
outband.WriteArray(img0)

# set coordinate reference system
outRasterSRS = osr.SpatialReference() # create instance of class
outRasterSRS.ImportFromEPSG(26911)    # set to specific coordinate reference system
# outRasterSRS.ImportFromWkt(raster.GetProjectionRef()) # or could use SRS from before
outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set the projection

# flush output to file
outband.FlushCache()

# this closes the files
raster = None
outRaster = None
# -
xxx


