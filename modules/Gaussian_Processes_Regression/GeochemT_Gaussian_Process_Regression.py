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

# + [markdown] editable=true slideshow={"slide_type": ""}
# # Geothermal and Machine Learning Sandbox

# + editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt

import math

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from scipy.ndimage import gaussian_filter

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

# %matplotlib inline

# + editable=true slideshow={"slide_type": ""}
resultsPath = './results/'

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## BEGIN data preprocessing
# -

# ### preprocessed data

# +
path = '../../datasets/'

filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

hf5File = path+filename
# -


# ### a geotiff base map to use for affine transforms

# +
path = '../../datasets/'

basemapFilename = 'BNN_(+)_q-0.05_avg-8192.tif'

basemapFile = path+basemapFilename
# -

# ### some reference maps

# +
path = '../../datasets/'

referencemapsFilename = 'PFA_reference_maps.h5'

referencemapsFile = path+referencemapsFilename
# -

# ### geochemistry temperature data

# +
path = '../../datasets/'

geochemTFilename = 'geochemT_training_sites.pkl'

geochemTFile = path+geochemTFilename
# -

# ### read data from files

# + editable=true slideshow={"slide_type": ""}
f = h5py.File(hf5File, 'r')

def keys(f):
    return [key for key in f.keys()]

key_list = keys(f)
print(key_list)

f.close()
# -

df_reference_maps = pd.read_hdf(referencemapsFile, 'df_reference_maps')

domain = df_reference_maps['domain'].copy().to_numpy()
domain = np.reshape(domain,(1000,-1))

hill = df_reference_maps['hillshade'].copy().to_numpy()
hill = np.reshape(hill,(1000,-1))

# # pare down info files and add target column

# ## get training site features for PCA/UMAP studies

dfn = pd.read_hdf(hf5File, key='dfn')
dfInfo = pd.read_hdf(hf5File, key='dfInfo')

# +
dfXyInfo = pd.read_hdf(hf5File, key='XyInfo')
dfX = pd.read_hdf(hf5File, key='X')
dfy = pd.read_hdf(hf5File, key='y')

df_posInfo = dfXyInfo[dfy == 1].copy()
df_negInfo = dfXyInfo[dfy == 0].copy()
df_posX = dfX[dfy == 1].copy()
df_negX = dfX[dfy == 0].copy()

df_posInfo['class'] = 'pos'
df_negInfo['class'] = 'neg'
# -

columns = dfn.columns.to_list()
# columns

# ## save heatflow map as a reference

hfMap = dfn['Heatflow'].copy().to_numpy()
hfMap = np.reshape(hfMap,(1000,-1))

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

# MASTER SET 1 - heatflow
################################# 0
['QuaternaryFaultTraces',
 'HorizGravityGradient2',
 'HorizMagneticGradient2',
 'GeodeticStrainRate',
 'QuaternarySlipRate',
 'FaultRecency',
 'FaultSlipDilationTendency2',
 'Earthquakes',
 # 'Heatflow',
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

# ## fix input types and dimensions

# #### entire study area

df_Xall = dfn[columns].copy()

# #### classification training sites

df_posX = df_posX[columns]
df_negX = df_negX[columns]

nFeatures = df_Xall.shape[1]
nFeatures

print (df_Xall.shape)

# ## read in geochemistry temperature dataframe

df_geoTRC = pd.read_pickle(geochemTFile)

df_geoTRC.reset_index(drop=True, inplace=True)

# ### there are multiple values per pixel
# ### ... find a typical value

df_geoTRC.columns

column = 'geochemT_final'

df_geoTRC['avg'] = df_geoTRC.groupby('id_rc')[column].transform('mean')
df_geoTRC['min'] = df_geoTRC.groupby('id_rc')[column].transform('min')
df_geoTRC['max'] = df_geoTRC.groupby('id_rc')[column].transform('max')

df_geoTRC['raw'] = df_geoTRC['avg'].copy()

# ## truncate first ?

truncate = True
quantile = 0.975

if truncate:

    fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1)
    
    data = df_geoTRC['raw'].values
    # data = df_HFRC['transformed'].values
    
    N = len(data)
            
    # sort the data in ascending order
    x = np.sort(data)
      
    # get the cdf values of y
    y = np.arange(N) / float(N)
    
    # idx = np.where(y>0.9)[0][0]
    idx = np.where(y>quantile)[0][0]
    print(idx, x[idx])
    xQuantile = x[idx]
    
    # plotting
    ax.plot(x[:], y[:], marker='o')
    ax.plot(x[idx], y[idx], color='red', marker='*')
    
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis') 
    ax.set_title('CDF using sorting the data')
    
    ax.grid(True)

    bigIndexes = df_geoTRC[df_geoTRC['raw'] > xQuantile].index
    df_geoTRC.drop(bigIndexes, inplace = True)

# ### the distribution is skewed
# ### ... transform to something more like a Gaussian

# +
from sklearn.preprocessing import PowerTransformer
pt_geochemT = PowerTransformer(standardize=True)

transformed = pt_geochemT.fit_transform(df_geoTRC['raw'].values.reshape(-1,1))
df_geoTRC['transformed'] = transformed

# print(pt_heatflow.lambdas_)
# -



# ### drop duplicates, i.e. get one value per pixel

df_geoTRC = df_geoTRC.drop_duplicates(subset='id_rc')

idrc = df_geoTRC['id_rc']

df_geoTRC_sorted = df_geoTRC.sort_values(by=['id_rc'])

df_geoTExtract = dfInfo[dfInfo['id_rc'].isin(idrc)].copy()

df_geoTExtract_sorted = df_geoTExtract.sort_values(by=['id_rc'])

df_geoTInfo = df_geoTExtract_sorted.iloc[:,:6].copy()
df_geoTInfo['class'] = 'geoT'

# +
raw = df_geoTRC_sorted['raw'].to_numpy()
transformed = df_geoTRC_sorted['transformed'].to_numpy()

df_geoTInfo['raw'] = raw
df_geoTInfo['transformed'] = transformed
# -
# ## set regression target

# +
# target = df_geoTRC_sorted['raw'].to_numpy()
target = df_geoTRC_sorted['transformed'].to_numpy()

df_geoTInfo['target'] = target
# -

df_geoTInfo = df_geoTInfo.sort_index()

# ### drop nulls

nullIndexes = df_geoTInfo[df_geoTInfo['NullInfo'] == 'nullValue'].index
df_geoTInfo.drop(nullIndexes, inplace = True)


df_geoTX = df_Xall.loc[df_geoTInfo.index].copy()

df_geoTy = df_geoTInfo['target']


df_geoTX.columns

df_geoTy.name

# +
fig, ax = plt.subplots(figsize=(12,4), nrows=1, ncols=2)

ax[0].hist(df_geoTRC['avg'].values, 100, density=True);
# ax[0].hist(df_geoTRC['max'].values, 100, density=True);
ax[1].hist(df_geoTRC['transformed'].values, 100, density=True);


# +
fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1)

data = df_geoTRC['transformed'].values

N = len(data)
        
# sort the data in ascending order
x = np.sort(data)
  
# get the cdf values of y
y = np.arange(N) / float(N)

idx = np.where(y>0.9)[0][0]
# idx = np.where(y>0.97)[0][0]
print(idx, x[idx])
xQuantile = x[idx]
# idx = N

# plotting
ax.plot(x[:], y[:], marker='o')

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis') 
ax.set_title('CDF using sorting the data')

ax.grid(True)
# -

# ## end selecting data



# +
# df_posInfo.columns.to_list()
# -

info_columns=['row', 'column', 'id_rc', 'X_83UTM11', 'Y_83UTM11', 'NullInfo', 'class']

# ## now concatenate for later use in PCA/UMAP views of feature space

df_infoClasses = pd.concat([ \
                            df_posInfo[info_columns],
                            df_negInfo[info_columns],
                            df_geoTInfo[info_columns],
                           ],
                           axis=0)


df_infoClasses.columns

df_XClasses = pd.concat([ \
                         df_posX,
                         df_negX,
                         df_geoTX,
                        ],
                        axis=0)

df_XClasses.columns

# +
X = df_XClasses.copy()
y = df_infoClasses['class'].copy()

yint = y.copy()
# -

yint.loc[(y=='neg')]=0
yint.loc[(y=='pos')]=1
yint.loc[(y=='geoT')]=-1

yint

y.loc[(y=='neg')]='n'
y.loc[(y=='pos')]='p'
y.loc[(y=='geoT')]='t'

y

# ## maybe need to random shuffle dataframes

Xs = X.reset_index(drop=True).copy()
ys = y.reset_index(drop=True).copy()
yints = yint.reset_index(drop=True).copy()

# +
# shuffle
# idx = np.random.permutation(Xs.index)

# don't shuffle
idx = Xs.index
# -

X = Xs.reindex(idx)
y = ys.reindex(idx)
yint = yints.reindex(idx)

df_embed = pd.DataFrame(X.copy())
df_embed['y'] = yint

# +
fig, ax = plt.subplots(figsize=(12,4))

ax.imshow(hill, cmap='gray', origin='lower')
ax.imshow(domain, cmap='gist_ncar', alpha=0.1, origin='lower')

ix = df_posInfo['column'].values
iy = df_posInfo['row'].values
ax.scatter(ix,iy, edgecolors='k', 
           marker='o', c='r', alpha=0.5,
           s=50)

ix = df_negInfo['column'].values
iy = df_negInfo['row'].values
ax.scatter(ix,iy, edgecolors='k', 
           marker='o', c='b', alpha=0.5,
           s=50)

ax.set(xticklabels=[]);
ax.set(yticklabels=[]);

ax.set_title('training sites')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='k', label='positive sites')
blue_patch = mpatches.Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='negative sites')

plt.legend(handles=[red_patch, blue_patch])

plt.savefig(resultsPath+'training_sites.png', dpi=300)



# +
fig, ax = plt.subplots(figsize=(12,4))

ax.imshow(hill, cmap='gray', origin='lower')
ax.imshow(domain, cmap='gist_ncar', alpha=0.1, origin='lower')

c=df_geoTInfo['target'].values

ix = df_geoTInfo['column'].values
iy = df_geoTInfo['row'].values

ax.scatter(ix,iy, edgecolors='k', 
           # marker='o', c='orange', alpha=0.5,
           marker='o', c=c, alpha=0.5,
           cmap='RdBu_r', s=50)

ax.set(xticklabels=[]);
ax.set(yticklabels=[]);

ax.set_title('geochemistry temperature sites')

plt.savefig(resultsPath+'geochemT_sites.png', dpi=300)

# -

# ### select large number of random sires for PCA/UMAP views
# ### ... then add these to the training sites

# +
nSamples = 10000

Xsample = df_Xall.copy()

Xsample = Xsample.sample(n=nSamples, random_state=42)

# get rid of nulls from edges of study area which are now zero features
Xsample_nullInfo = dfInfo.loc[Xsample.index, 'NullInfo']
nullIndexes = Xsample[Xsample_nullInfo == 'nullValue'].index
Xsample.drop(nullIndexes, inplace = True)

Xpos   = df_posX.copy()
Xneg   = df_negX.copy()
XgeoT    = df_geoTX.copy()
# -


# #### append regression training sites

Xsample = pd.concat([Xsample, Xpos], axis=0, join='outer')
Xsample = pd.concat([Xsample, Xneg], axis=0, join='outer')
# Xsample = pd.concat([Xsample, XgeoT], axis=0, join='outer')
ysample = df_reference_maps['domain'].copy().loc[Xsample.index]


df_embed = pd.DataFrame(Xsample.copy())


# ## PCA

from sklearn.decomposition import PCA


# +
pca = PCA(n_components=2)

pcaFit = pca.fit(Xsample.to_numpy())

# embedX = reducer.transform(X.copy())
embedX = pcaFit.transform(Xsample.to_numpy().copy())

print (embedX.shape)
# -


df_embed['pcaX'] = embedX[:,0]
df_embed['pcaY'] = embedX[:,1]

# ## data exploration

# ### plot reduced features w/o labels

# +
fig, ax = plt.subplots(figsize=(6,6))

points = ax.scatter(df_embed['pcaX'], df_embed['pcaY'], edgecolors='k', 
                    # marker='o', c='k', alpha=0.5,
                    marker='o', c='gray', alpha=0.1,
                    s=20)

ax.axis('square')
ax.grid(True)

# plt.savefig(resultsPath+'PCA_clusters_nolabels.png', dpi=300)
# -

# ### plot reduced features colored by domains



# +
Xpos = pcaFit.transform(df_posX.to_numpy())
Xneg = pcaFit.transform(df_negX.to_numpy())

XgeoT = pcaFit.transform(df_geoTX[df_geoTInfo['target']>xQuantile])

# +
fig, ax = plt.subplots(figsize=(6,6))

points = ax.scatter(df_embed['pcaX'], df_embed['pcaY'], edgecolors='k', 
                    # marker='o', c='k', alpha=0.5,
                    marker='o', c=ysample, alpha=0.1,
                    s=20, cmap='gist_ncar')

points0 = ax.scatter(Xpos[:,0], Xpos[:,1], edgecolors='k', 
                    marker='o', c='r', alpha=0.5,
                    s=30)

points1 = ax.scatter(Xneg[:,0], Xneg[:,1], edgecolors='k', 
                    marker='o', c='b', alpha=0.5,
                    s=30)

points2 = ax.scatter(XgeoT[:,0], XgeoT[:,1], edgecolors='k', 
                    marker='o', c='yellow', alpha=1,
                    s=30)

ax.axis('square')
ax.grid(True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='k', label='positive sites')
blue_patch = mpatches.Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='negative sites')
yellow_patch = mpatches.Patch(facecolor='yellow', alpha=0.5, edgecolor='k', label='high T')

plt.legend(handles=[red_patch, blue_patch, yellow_patch])

plt.savefig(resultsPath+'geochemT_PCA_clusters.png', dpi=300)

# -
# ## UMAP

import umap as UMAP

# +
umap = UMAP.UMAP(random_state=42, 
                    min_dist=0.01, 
                    n_neighbors=600)

umapFit = umap.fit(Xsample.copy())

embedX = umapFit.transform(Xsample.copy())
                            
embedX.shape
# -

df_embed['umapX'] = embedX[:,0]
df_embed['umapY'] = embedX[:,1]

# +
Xpos = umapFit.transform(df_posX.to_numpy())
Xneg = umapFit.transform(df_negX.to_numpy())

XgeoT = umapFit.transform(df_geoTX[df_geoTInfo['target']>xQuantile])
# -

# ### plot reduced features and color by domains and add training sites
# ### ... goal is to have reduced feature space that embodies all training sites

# +
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6))

points = ax.scatter(df_embed['umapX'], df_embed['umapY'], edgecolors='k', 
                    marker='o', c=ysample, alpha=0.1,
                    # marker='o', c='deepskyblue',
                    s=20, cmap="gist_ncar")

points0 = ax.scatter(Xpos[:,0], Xpos[:,1], edgecolors='k', 
                    marker='o', c='r', alpha=0.5,
                    s=30)

points1 = ax.scatter(Xneg[:,0], Xneg[:,1], edgecolors='k', 
                    marker='o', c='b', alpha=0.5,
                    s=30)

points2 = ax.scatter(XgeoT[:,0], XgeoT[:,1], edgecolors='k', 
                    marker='o', c='yellow', alpha=1,
                    s=30)

ax.axis('square')
ax.grid(True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(facecolor='red', alpha=0.5, edgecolor='k', label='positive sites')
blue_patch = mpatches.Patch(facecolor='blue', alpha=0.5, edgecolor='k', label='negative sites')
yellow_patch = mpatches.Patch(facecolor='yellow', alpha=0.5, edgecolor='k', label='high T')

plt.legend(handles=[red_patch, blue_patch, yellow_patch])

plt.savefig(resultsPath+'geochemT_umap_clusters.png', dpi=300)



# +
# xxx
# -

# ## k-means clustering

from sklearn.cluster import KMeans

model = KMeans(n_clusters=6, random_state=42, n_init='auto')

# +
# Xk = df_embed[['pcaX', 'pcaY']].copy()
Xk = df_embed[['umapX', 'umapY']].copy()

labels_true = ysample.copy()
# labels_true = df_embed['y'].copy()
# -

model.fit(Xk)

model.predict(Xk[:10])

centroids = model.cluster_centers_
centroids

# +
fig, ax = plt.subplots(figsize=(6,6))

# points = ax.scatter(df_embed['umapX'], df_embed['umapY'], edgecolors='k', 
points = ax.scatter(Xk.iloc[:,0], Xk.iloc[:,1], edgecolors='k', 
                    marker='o', c=ysample, alpha=0.1,
                    # marker='o', c='gray', alpha=0.1,
                    # marker='o', c=df_embed['y'], alpha=0.7,
#                     marker='o', c='deepskyblue',
                    s=20, cmap="gist_ncar")

ax.axis('square')
ax.grid(True)

ax.scatter(centroids[:,0], centroids[:,1], 
           edgecolors='k', marker='*',
           s=200, color='yellow')
# -
# ### can use model.predict(Xk) or model.labels_ to see which cluster each point belongs to and change the symbol or color to mark them

model.predict(Xk)

# ### Bayesian Gaussian Mixture

from sklearn.mixture import BayesianGaussianMixture

# +
# Xk = df_embed[['pcaX', 'pcaY']].copy()
Xk = df_embed[['umapX', 'umapY']].copy()

labels_true = ysample.copy()
# labels_true = df_embed['y'].copy()
# -

model = BayesianGaussianMixture(n_components=6, random_state=42, max_iter=500)

model.fit(Xk)

preds = model.predict(Xk)
preds = pd.DataFrame(preds)
preds.index = Xk.index

preds.columns = ['clusters']

probs = model.predict_proba(Xk)
probs = pd.DataFrame(probs)
probs.index = Xk.index

cols = []
for c in probs.columns:
    cols.append('p'+str(c))
probs.columns = cols

BGMM = Xk.iloc[:,0:2].copy()
BGMM = pd.concat([BGMM, preds], axis=1)
BGMM = pd.concat([BGMM, probs], axis=1)

means = model.means_

covars = model.covariances_

weights = model.weights_

# +
import matplotlib as mpl

def plot_ellipses(ax, alphas, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], 
                                  width=eig_vals[0], 
                                  height=eig_vals[1],
                                  angle=180+angle, 
                                  edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alphas[n])
        # ell.set_facecolor('#56B4E9')
        ell.set_facecolor('gray')
        ellArtist = ax.add_artist(ell)
        ellArtist.set_zorder(0)

# +
# centroids = centroids[(1,2),:]

# +
fig, ax = plt.subplots(figsize=(6,6))

nSigma = 1
alphas = 2*weights

plot_ellipses(ax, alphas, means, nSigma*covars)

pointsp = ax.scatter(Xk.iloc[:,0], Xk.iloc[:,1], edgecolors='k', 
                    # marker='o', c=ysample, alpha=0.1,
                    marker='o', c=BGMM['p0'], alpha=0.1,
                    # marker='o', c=BGMM['clusters'], alpha=0.1,
#                     marker='o', c='deepskyblue',
                    # s=10, cmap="gist_ncar")
                    s=10, cmap="RdBu_r")

ax.axis('square')
ax.grid(True)

ax.scatter(means[:,0], means[:,1], 
           edgecolors='k', marker='*',
           s=100, color='yellow')

for i in range(means.shape[0]):
    plt.text(means[i,0]+0.1,means[i,1]+0.1, str(i), fontsize=14, fontweight='roman', color='k')

g_patch = mpatches.Patch(color='g', label='paleo', alpha=0.5)
b_patch = mpatches.Patch(color='blue', label='neg', alpha=0.5)
r_patch = mpatches.Patch(color='red', label='pos', alpha=0.5)
gray_patch = mpatches.Patch(color='gray', label=str(nSigma)+'$\sigma$ BGMM cluster', alpha=0.5)

plt.legend(handles=[gray_patch])

# +
dfRC = dfInfo.loc[BGMM.index]

dfRC = dfRC[['row', 'column']]
# -


BGMM = pd.concat([BGMM, dfRC], axis=1)

subBGMM0 = BGMM[BGMM['clusters']==0]
subBGMM1 = BGMM[BGMM['clusters']==1]
subBGMM2 = BGMM[BGMM['clusters']==2]
subBGMM3 = BGMM[BGMM['clusters']==3]
subBGMM4 = BGMM[BGMM['clusters']==4]
subBGMM5 = BGMM[BGMM['clusters']==5]

# +
fig, ax = plt.subplots(figsize=(12,4))

ax.imshow(hill, cmap='gray', origin='lower')
ax.imshow(domain, cmap='gist_ncar', alpha=0.1, origin='lower')

ix = subBGMM5['column'].values
iy = subBGMM5['row'].values
c = subBGMM5['clusters']

ax.scatter(ix,iy, edgecolors=None, 
           marker='o', c='blue', alpha=0.1,
           # s=50, cmap="coolwarm")
           s=20)

ax.set(xticklabels=[]);
ax.set(yticklabels=[]);


# +
fig, ax = plt.subplots(figsize=(10,4))

cluster = 0

ix = BGMM['column'].values
iy = BGMM['row'].values
# c = np.exp(BGMM['p5'])
c = BGMM['clusters']
a = BGMM['p'+str(cluster)].copy()

a[a<0.5] = 0.0
a /= 5

ax.scatter(ix,iy, edgecolors='k', 
           marker='o', c=c, alpha=a,
           # s=20, cmap="Blues")
           s=20, cmap='gist_ncar')
           # s=20, cmap='RdBu_r')

ax.imshow(hill, cmap='gray', alpha=0.95, origin='lower')
ax.imshow(domain, cmap='gist_ncar', alpha=0.2, origin='lower')

ax.set(xticklabels=[]);
ax.set(yticklabels=[]);
# -


# # scikits-learn gaussian processes regression

# +
# xxx
# -

# ## Here define regression data set

problem = 'geochemistry temperature'

X = df_geoTX
y = df_geoTy

inferX0 = umapFit.transform(X.copy().to_numpy())


# ### fit all features or only the reduced (e.g. UMAP) features

# +
# fit reduced features
# XtrainR = inferX0

# fit all features
XtrainR = X.copy()

ytrainR = y.copy()
# -

XtrainR.shape

ytrainR.shape

# +
plt.scatter(df_embed['umapX'], df_embed['umapY'], edgecolors=None, 
                    marker='s', c='lightgray', alpha=0.1,
                    s=100)

plt.scatter(inferX0[:,0], inferX0[:,1], edgecolors=None, 
                    # marker='o', c=ytrain, alpha=0.1,
                    marker='o', c=ytrainR, alpha=0.5,
                    # marker='o', c='deepskyblue',
                    # s=20, cmap="gist_ncar")
                    s=50, cmap="RdBu_r")


plt.axis('square')
plt.grid(True)

plt.colorbar()

plt.suptitle(problem+' transformed training set')
plt.title('      in umap reduced dimensions')

# -
# ## guess of kernel scale parameter

# +
s = np.std(ytrainR) / (ytrainR.max()-ytrainR.min())

s**2

# +
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, DotProduct, WhiteKernel

# kernel = DotProduct() + WhiteKernel()

# kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
#     DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 3
# )

# )


# This works with 10 features reduced to 2 umap
# kernel = 10.0 * RBF(length_scale=1, length_scale_bounds=(0.1, 1)) + \
#                 WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-5, 5e3))

# this works with 9 features reduced to 2 umap
# kernel = 10.0 * RBF(length_scale=0.3, length_scale_bounds=(0.1, 0.5)) + \
#                 WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-5, 5e3))

# this works with 10 features
# kernel = 10.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + \
#                  WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))

kernel = ConstantKernel(s**2, constant_value_bounds="fixed") * \
         RBF(length_scale=1.5, length_scale_bounds="fixed")+ \
         WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
         # WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")

        # WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
         # WhiteKernel(noise_level=0.00001, noise_level_bounds="fixed")
         # RBF(length_scale=2, length_scale_bounds="fixed")+ \
         # WhiteKernel(noise_level=0.5, noise_level_bounds="fixed")
         # WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1))

gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0).fit(XtrainR, ytrainR)

gpr.score(XtrainR, ytrainR)

# gpr.predict(Xsample[:2,:], return_std=True)
# -


# ### depending on if we fit all features or only the reduced (e.g. UMAP) features

# +
# ypred, ystd = gpr.predict(df_embed[['umapX','umapY']].to_numpy(), return_std=True)
# ypred, ystd = gpr.predict(df_embed[['pcaX','pcaY']].to_numpy(), return_std=True)

ypred, ystd = gpr.predict(Xsample, return_std=True)
# -

ypred.shape

# +
# ypreds = gpr.sample_y(df_embed[['umapX','umapY']].to_numpy())

# +
fig,ax = plt.subplots(figsize=(12,4), nrows=1, ncols=2)

# plt.scatter(df_embed['pcaX'], df_embed['pcaY'], edgecolors=None, 
p0 = ax[0].scatter(df_embed['umapX'], df_embed['umapY'], edgecolors=None, 
                    marker='o', c=ypred, alpha=1,
                    # s=10, cmap="RdBu_r")
                    s=10, cmap="RdBu_r")
                    # s=20, cmap="gist_ncar")
                    # s=20, cmap="coolwarm")
    
ax[0].axis('square')
ax[0].grid(True)
plt.colorbar(p0,label='heatflow transformed residual')


# plt.scatter(df_embed['pcaX'], df_embed['pcaY'], edgecolors=None, 
p1 = ax[1].scatter(df_embed['umapX'], df_embed['umapY'], edgecolors=None, 
                    marker='o', c=ystd, alpha=1,
                    s=10, cmap="RdBu_r")
                    # s=20, cmap="gist_ncar")
                    # s=20, cmap="coolwarm")

ax[1].axis('square')
ax[1].grid(True)
plt.colorbar(p1,label=problem+' transformed stddev')

plt.suptitle(problem+' regression in umap reduced dimensions');

plt.tight_layout()
# -


# ### invert transformed values

# +
predInverse = pt_geochemT.inverse_transform(ypred.reshape(-1,1))
# stdInv = pt_heatflow.inverse_transform(ystd.reshape(-1,1))

df_regression = dfRC.copy()
df_regression['predInverse'] = predInverse
df_regression['predTransformed'] = ypred
df_regression['stdTransformed'] = ystd

# df_regression['stdInv'] = stdInv


# +
from sklearn.neighbors import KernelDensity

bins = np.linspace(-5, 5, 50)
X_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]

kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(ytrainR.to_numpy().reshape(-1, 1))
log_dens = kde.score_samples(X_plot)

kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(ypred.reshape(-1, 1))
log_dens_pred = kde_pred.score_samples(X_plot)


# +
plt.hist(ytrainR, bins=bins, density=True, alpha=0.2);
plt.hist(ypred, 50, density=True, alpha=0.2);
plt.plot(X_plot, np.exp(log_dens))
plt.plot(X_plot, np.exp(log_dens_pred))
# plt.plot(X_plot, (log_dens))
# plt.plot(X_plot, (log_dens_pred))

# plt.ylim(0,1)
# -

# ## KL divergence
#
# https://machinelearningmastery.com/divergence-between-probability-distributions/

# +
from scipy.special import rel_entr, kl_div

# define distributions
p = np.exp(log_dens)
q = np.exp(log_dens_pred)

# calculate (P || Q)
kl_pq = rel_entr(p, q)
print('KL(P || Q): %.3f nats' % np.sum(kl_pq))
# calculate (Q || P)
kl_qp = rel_entr(q, p)
print('KL(Q || P): %.3f nats' % np.sum(kl_qp))

# calculate (P || Q)
kl = kl_div(p, q)
print('KL(P || Q): %.3f nats' % np.sum(kl))
# calculate (Q || P)
kl = kl_div(q, p)
print('KL(Q || P): %.3f nats' % np.sum(kl))


# calculate the kl divergence
def kl_divergence(p, q):
    s = np.sum(p * np.log2(p/q))
    return s               
    # return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

kl_pq = rel_entr(p, q)
print('KL(P || Q): %.3f nats' % np.sum(kl_pq))
# calculate (Q || P)
kl_qp = rel_entr(q, p)
print('KL(Q || P): %.3f nats' % np.sum(kl_qp))


# calculate the js divergence
def js_divergence(p, q):
 m = 0.5 * (p + q)
 return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# calculate JS(P || Q)
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % np.sqrt(js_pq))

# calculate the jensen-shannon distance metric
from scipy.spatial.distance import jensenshannon
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)
# calculate JS(Q || P)
js_qp = jensenshannon(q, p, base=2)
print('JS(Q || P) Distance: %.3f' % js_qp)
# -

# ### plot maps of predicted values for our PCA/UMAP random samples + training sites

# +
fig, ax = plt.subplots(2,1, figsize=(12,8))

##############################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['predInverse']

a = 1

ax[0].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[0].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           s=5, cmap='RdBu_r', vmin=0, vmax=None)
           # s=5, cmap='RdBu_r', vmin=-350, vmax=350)
           # s=5, cmap='RdBu_r', vmin=-350, vmax=350)

# ax.imshow(domain, cmap='gray', alpha=0.8, origin='lower')

ax[0].set(xticklabels=[]);
ax[0].set(yticklabels=[]);

ax[0].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='predicted '+problem)

##############################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['predTransformed']

a = 1

ax[1].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[1].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           # s=5, cmap='RdBu_r', vmin=-200, vmax=200)
           s=5, cmap='RdBu_r', vmin=-6, vmax=6)

# ax.imshow(domain, cmap='gray', alpha=0.8, origin='lower')

ax[1].set(xticklabels=[]);
ax[1].set(yticklabels=[]);

ax[1].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='predicted transformed '+problem)

# +
fig, ax = plt.subplots(2,1, figsize=(12,8))

##############################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['predTransformed']

a = 1

ax[0].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[0].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           # s=5, cmap='RdGy_r')
           s=5, cmap='RdBu_r', vmin=-6, vmax=6)

# ax.imshow(domain, cmap='gray', alpha=0.8, origin='lower')

ax[0].set(xticklabels=[]);
ax[0].set(yticklabels=[]);

ax[0].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='predicted transformed '+problem)

##############################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['stdTransformed']
# c = c-c.mean()
# c = df_regression['stdTransformed']/(np.mean(df_regression['predTransformed']))
  
a = 1

ax[1].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[1].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           # s=5, cmap='Blues_r', vmin=0, vmax=0.15)
           # s=5, cmap='coolwarm', vmin=0, vmax=0.15)
           # s=5, cmap='Blues', vmin=0, vmax=0.1)
           # s=5, cmap='Blues')
           s=5, cmap='RdBu_r')
           # s=5, cmap='Reds', vmin=0, vmax=1)
           # s=5, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
           # s=5, cmap='coolwarm')

           # s=5, cmap='RdBu_r', vmin=0, vmax=0.1)

# ax.imshow(domain, cmap='gray', alpha=0.8, origin='lower')

ax[1].set(xticklabels=[]);
ax[1].set(yticklabels=[]);

ax[1].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='std transformed '+problem)

# -
plt.hist(df_regression['stdTransformed']/np.mean(df_regression['predTransformed']),50);
# plt.hist(np.abs(df_regression['stdTransformed']),50);
# plt.hist(np.abs(df_regression['predTransformed']),50);

# +
fig, ax = plt.subplots(2,1, figsize=(12,8))

#################################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['predInverse']

a = 1

ax[0].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[0].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           # s=5, cmap='RdGy_r', vmin=-3, vmax=3)
           s=2, cmap='RdGy_r', vmin=-350, vmax=350)

c = ax[0].contour(hfMap, cmap='RdBu_r', alpha=1, origin='lower')

ax[0].set(xticklabels=[]);
ax[0].set(yticklabels=[]);

ax[0].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='predicted transformed '+problem)

# plt.clabel(c, inline=1, fontsize=10)
h,l = c.legend_elements()
plt.legend([h[-1]], ['heatflow'])

#################################################################
# plt.subplot(122)
c = ax[1].imshow(hfMap, cmap='RdBu_r', alpha=0.5, origin='lower')

ax[1].set(xticklabels=[]);
ax[1].set(yticklabels=[]);

ax[1].set_title('PFA study area')

plt.colorbar(c, label='scaled heatflow')

# +
fig, ax = plt.subplots(2,1, figsize=(12,8))

#################################################################
ix = df_regression['column'].values
iy = df_regression['row'].values
c = df_regression['predTransformed']

a = 1

ax[0].imshow(hill, cmap='gray', alpha=0.01, origin='lower')
# ax.imshow(domain, cmap='gray', alpha=1, origin='lower')

pts = ax[0].scatter(ix,iy, edgecolors=None, 
           marker='s', c=c, alpha=a,
           # s=5, cmap='RdGy_r', vmin=-3, vmax=3)
           s=2, cmap='RdGy_r', vmin=-6, vmax=6)
           # s=5, cmap='RdGy_r', vmin=-350, vmax=350)

c = ax[0].contour(hfMap, cmap='RdBu_r', alpha=1, origin='lower')

ax[0].set(xticklabels=[]);
ax[0].set(yticklabels=[]);

ax[0].set_title('PFA study area '+problem+' regression')

plt.colorbar(pts, label='predicted transformed '+problem)

# plt.clabel(c, inline=1, fontsize=10)
h,l = c.legend_elements()
plt.legend([h[-1]], ['heatflow'])

#################################################################
# plt.subplot(122)
c = ax[1].imshow(hfMap, cmap='RdBu_r', alpha=0.5, origin='lower')

ax[1].set(xticklabels=[]);
ax[1].set(yticklabels=[]);

ax[1].set_title('PFA study area')

plt.colorbar(c, label='scaled heatflow')

# +
# xxx
# -

# ## prediction for every pixel
# ### use pytorch dataloaders for convenience

df_Xall.columns

# +
Xall = df_Xall.copy()
Xall = Xall.to_numpy()

columns = df_Xall.columns
# -

# ## develop image by sampling

# +
###########################################
t0 = time.time()
###########################################

# inference in batches
eval_loader = DataLoader(Xall, shuffle=False, 
                         batch_size=512, drop_last=False)

# yResultMean = []
yResult = []
yResultStd = []
yResultInv = []
yResultStdInv = []

# break image into batches
for features in tqdm(eval_loader):

    # batch of features
    Xsample = pd.DataFrame(features.numpy(), columns=columns)    

    # can directly get mean and stddev of GP
    #   the mean is exact and should be used, but stddev seems truncated 
    #   ... will use stddev of samples instead
    # ypredMean, ypredStd = gpr.predict(Xsample, return_std=True)
    ypredMean = gpr.predict(Xsample, return_std=False)
    ypredMean_shape = ypredMean.shape

    # sample the Gaussian Process for each pixel in a batch
    ###########################################
    ysample = gpr.sample_y(Xsample, n_samples=1024)
    sample_shape = ysample.shape    

    ypredMeanInverse = pt_geochemT.inverse_transform(ypredMean.reshape(-1,1))
    ypredMeanInverse = ypredMeanInverse.reshape(ypredMean_shape)
    
    ysampleInverse = pt_geochemT.inverse_transform(ysample.reshape(-1,1))
    ysampleInverse = ysampleInverse.reshape(sample_shape)
    ###########################################

    # compute the statistics for each pixel in a batch
    ###########################################
    ypred = ypredMean
    # ypred = np.mean(ysample, axis=1)
    ystd = np.std(ysample, axis=1)
    
    ypredInverse = ypredMeanInverse
    # ypredInverse = np.mean(ysampleInverse, axis=1)
    ystdInverse = np.std(ysampleInverse, axis=1)
    ###########################################
    
    # append the batches together
    ###########################################
    # ypredMean = Tensor(ypredMean).squeeze()
    ypred = Tensor(ypred).squeeze()
    ystd = Tensor(ystd).squeeze()
    ypredInverse = Tensor(ypredInverse).squeeze()
    ystdInverse = Tensor(ystdInverse).squeeze()

    # yResultMean.append(ypredMean)
    yResult.append(ypred)
    yResultStd.append(ystd)
    yResultInv.append(ypredInverse)
    yResultStdInv.append(ystdInverse)
    ###########################################

# reassemble the image
###########################################
# yResultMean = torch.cat(yResultMean, dim=0)
yResult = torch.cat(yResult, dim=0)
yResultStd = torch.cat(yResultStd, dim=0)
yResultInv = torch.cat(yResultInv, dim=0)
yResultStdInv = torch.cat(yResultStdInv, dim=0)
###########################################


###########################################
print ('Inference time = ', time.time()-t0)
###########################################

print (yResult.shape)
# -

# ## plot results

# +
# img = yResult.numpy().reshape(1000,-1)
img = yResultInv.numpy().reshape(1000,-1)
img = np.flipud(img)

# imgstd = yResultStd.numpy().reshape(1000,-1)
imgstd = yResultStdInv.numpy().reshape(1000,-1)
imgstd = np.flipud(imgstd)
# -

img.shape

# +
fig, ax = plt.subplots(2,1, figsize=(8,8))

plt.suptitle('Gaussian Process Regression')
# plt.suptitle('GP sampled')
# plt.suptitle('GP inverse cheat')

# plt.imshow(img, cmap='RdBu_r', vmin=0, vmax=None)
p0 = ax[0].imshow(img, cmap='RdBu_r')
plt.colorbar(p0, label='predicted '+problem) 
    
# p1 = ax[1].imshow(imgstd, cmap='RdBu_r')
p1 = ax[1].imshow(imgstd, cmap='coolwarm')
# p1 = ax[1].imshow(imgstd, cmap='RdBu_r', vmin=0, vmax=0.2)

# plt.colorbar(p1, label='predicted stddev transformed '+problem)
plt.colorbar(p1, label='predicted stddev '+problem)

plt.tight_layout()

# plt.savefig(resultsPath+'geochemT_transformed_regression_prediction_maps.png')
plt.savefig(resultsPath+'geochemT_invtransformed_regression_prediction_maps.png')

# plt.savefig('sampled_regression_prediction_maps.png')
# plt.savefig('GP_inverse_cheat_regression_prediction_maps.png')

# +
fig, ax = plt.subplots(1,1, figsize=(8,4))

plt.suptitle('Gaussian Process Regression')
# plt.suptitle('GP sampled')
# plt.suptitle('GP inverse cheat')

# plt.imshow(img, cmap='RdBu_r', vmin=0, vmax=None)
p0 = ax.imshow(img, cmap='RdBu_r')
plt.colorbar(p0, label='predicted '+problem) 
    
# # p1 = ax[1].imshow(imgstd, cmap='RdBu_r')
# p1 = ax[1].imshow(imgstd, cmap='coolwarm')
# # p1 = ax[1].imshow(imgstd, cmap='RdBu_r', vmin=0, vmax=0.2)

# # plt.colorbar(p1, label='predicted stddev transformed '+problem)
# plt.colorbar(p1, label='predicted stddev '+problem)

plt.tight_layout()

# plt.savefig(resultsPAth+'geochemT_transformed_regression_prediction_maps_0.png')
plt.savefig(resultsPath+'geochemT_invtransformed_regression_prediction_maps_0.png', dpi=300)

# plt.savefig('transformed_regression_prediction_maps.png')
# plt.savefig('sampled_regression_prediction_maps.png')
# plt.savefig('GP_inverse_cheat_regression_prediction_maps.png')
# -

plt.hist(img.ravel(), 50);

plt.hist(imgstd.ravel(), 50);

xxx

# ## save results to pickle file

# +
frame = {'geoT_transformed':yResult, 'geoT_transformed_std':yResultStd,
         'geoT':yResultInv, 'geoT_std':yResultStdInv}

df_results = pd.DataFrame(frame)
# -

df_results

info_columns

dfInfo[info_columns[:-1]]

df_HF_residual_GPRregression = pd.concat([dfInfo[info_columns[:-1]], df_results],axis=1)

# +
results_dict = {'df_geoT_GPRegression':df_HF_residual_GPRregression,
                'df_XClasses':df_XClasses, 
                'df_infoClasses':df_infoClasses}

with open(resultsPath+'geochemT_GPRegression.pkl', 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -

# ## check pickle file

# +
with open(resultsPath+'geochemT_GPRegression.pkl', 'rb') as handle:
    b = pickle.load(handle)

b.keys()
# -

b['df_geoT_GPRegression']

b['df_geoT_GPRegression'].columns



# +
# df.to_hdf('HF_residual_GPRegression.h5', key='df_HF_residial_GPRegression', mode='w') 
# df_read = pd.read_hdf('HF_residual_GPRegression.h5', 'df_HF_residial_GPRegression')
# -

xxx





# # $\color{red}{\text{continue editing here}}$





# # Explore some ways to optimize GP parameters

# # cross validation

# +
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, DotProduct, WhiteKernel

# kernel = DotProduct() + WhiteKernel()

# kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
#     DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 3
# )

# )


# This works with 10 features reduced to 2 umap
# kernel = 10.0 * RBF(length_scale=1, length_scale_bounds=(0.1, 1)) + \
#                 WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-5, 5e3))

# this works with 9 features reduced to 2 umap
# kernel = 10.0 * RBF(length_scale=0.3, length_scale_bounds=(0.1, 0.5)) + \
#                 WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-5, 5e3))

# # this works with 10 features
# kernel = 10.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + \
#                  WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))


# gpr = GaussianProcessRegressor(kernel=kernel,
#         random_state=0).fit(XtrainR, ytrainR)

kernel = ConstantKernel(s**2, constant_value_bounds="fixed") * \
         RBF(length_scale=2, length_scale_bounds="fixed")+ \
         WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))

gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0).fit(XtrainR, ytrainR)

gpr.score(XtrainR, ytrainR)

# gpr.predict(Xsample[:2,:], return_std=True)


# +
# df_embed[['umapX','umapY']]

# +
# ypred, ystd = gpr.predict(df_embed[['umapX','umapY']].to_numpy(), return_std=True)
# ypred, ystd = gpr.predict(df_embed[['tsneX','tsneY']].to_numpy(), return_std=True)
# ypred, ystd = gpr.predict(df_embed[['pcaX','pcaY']].to_numpy(), return_std=True)

ypred, ystd = gpr.predict(Xsample, return_std=True)
# -



# +
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=2)
# for train, test in kf.split(XtrainR):
#     print("%s %s" % (train, test))



# +
from sklearn.model_selection import ShuffleSplit

kernel = ConstantKernel(s**2, constant_value_bounds="fixed") * \
         RBF(length_scale=2, length_scale_bounds="fixed")+ \
         WhiteKernel(noise_level=0.5, noise_level_bounds="fixed")

gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0)

ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

scoresTrain = []
scoresTest = []
for train_index, test_index in ss.split(X):
    gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
    scoresTrain.append(gpr.score(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index]))
    scoresTest.append(gpr.score(XtrainR.iloc[test_index,:], ytrainR.iloc[test_index]))

print (np.mean(scoresTrain), np.mean(scoresTest))

# scores = []
# for train_index, test_index in ss.split(XtrainR):
#     gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
#     scores.append(gpr.score(XtrainR.iloc[test_index,:], ytrainR.iloc[test_index]))

# print (np.mean(scores))

# print(scores)

# gpr.fit(XtrainR, ytrainR)
# gpr.score(XtrainR, ytrainR)

# -

kernel

# +
from sklearn.model_selection import KFold

# # this works with 10 features
# kernel = 10.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + \
#                  WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))

kernel = ConstantKernel(s**2, constant_value_bounds="fixed") * \
         RBF(length_scale=2, length_scale_bounds="fixed")+ \
         WhiteKernel(noise_level=0.5, noise_level_bounds="fixed")

gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0)

X = XtrainR
kf = KFold(n_splits=10)

scoresTrain = []
scoresTest = []
for train_index, test_index in kf.split(X):
    gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
    scoresTrain.append(gpr.score(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index]))
    scoresTest.append(gpr.score(XtrainR.iloc[test_index,:], ytrainR.iloc[test_index]))
    
    

print (np.mean(scoresTrain), np.mean(scoresTest))

# print(scores)

# -



# +
from sklearn.model_selection import LeaveOneOut, LeavePOut

loo = LeaveOneOut()
# lpo = LeavePOut(10)

kernel = ConstantKernel(s**2, constant_value_bounds="fixed") * \
         RBF(length_scale=2, length_scale_bounds="fixed")+ \
         WhiteKernel(noise_level=0.5, noise_level_bounds="fixed")

gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0)

X = XtrainR

scoresTrain = []
scoresTest = []
for train_index, test_index in loo.split(X):
    gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])   
    errorTrain = np.mean((gpr.predict(XtrainR.iloc[train_index,:]) - ytrainR.iloc[train_index])**2)
    errorTest  = np.mean((gpr.predict(XtrainR.iloc[test_index,:]) - ytrainR.iloc[test_index])**2)
    scoresTrain.append(errorTrain)
    scoresTest.append(errorTest)

print (np.mean(scoresTrain), np.mean(scoresTest))

# print(scores)

# -



parRange = [np.arange(1, 5, 2), np.arange(0.1, 2, 0.5)]
parRange[1].shape

# ## loop over parameters

# +
from sklearn.model_selection import KFold
import itertools


##################################################################
# include kl divergence as metric
##################################################################
from sklearn.neighbors import KernelDensity
bins = np.linspace(-5, 5, 50)
X_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]

from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
##################################################################

# parRange = [np.arange(1, 20, 2), np.arange(0.0, 1, 0.5)]
# parRange = [np.arange(1, 5, 1), np.logspace(-1,3,10)]
# parRange = [np.logspace(np.log10(1), np.log10(5), 10), np.logspace(-1,0,10)]
# parRange = [np.linspace(1, 5, 9), np.logspace(-1,0,10)]
# parRange = [np.linspace(1, 5, 5), np.logspace(-1,0,5)]
# parRange = [np.linspace(0.1, 10, 9), np.logspace(-1,1.2,10)]
parRange = [np.logspace(np.log10(0.2), np.log10(20), 10), np.logspace(-1,1.2,20)]
# parRange = [np.linspace(1, 5, 4),np.logspace(-1,0, 4)]

kernels = [ConstantKernel(s**2, constant_value_bounds="fixed") * \
           RBF(length_scale=a, length_scale_bounds="fixed")+ \
           WhiteKernel(noise_level=b, noise_level_bounds="fixed") \
           for a, b in list(itertools.product(*parRange))]

results = []
for kernel in kernels:
    gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0)

    X = XtrainR
    kf = KFold(n_splits=10)

    scoresTrain = []
    errorsTrain = []
    errorsTest = []
    stdsTrain = []
    stdsTest = []
    kl = np.array([])
    for train_index, test_index in kf.split(X):
        gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
        
        scoreTrain = gpr.score(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
        scoresTrain.append(scoreTrain)
        
        mean, std = gpr.predict(XtrainR.iloc[train_index,:], return_std=True)
        yTrain = ytrainR.iloc[train_index]
        errorTrain = np.mean((mean-yTrain)**2)
        stdTrain = np.mean(std)
        errorsTrain.append(errorTrain)
        stdsTrain.append(stdTrain)
        
        mean, std = gpr.predict(XtrainR.iloc[test_index,:], return_std=True)
        yTest = ytrainR.iloc[test_index]
        errorTest = np.mean((mean-yTest)**2)
        stdTest = np.mean(std)
        errorsTest.append(errorTest)
        stdsTest.append(stdTest)
                    
    ##################################################################
    # include kl divergence as metric
    ##################################################################
    kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(ytrainR.to_numpy().reshape(-1, 1))
    log_dens = kde.score_samples(X_plot)

    mean, std = gpr.predict(XtrainR, return_std=True)
    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(mean.reshape(-1, 1))
    log_dens_pred = kde_pred.score_samples(X_plot)

    # define distributions
    p = np.exp(log_dens)
    q = np.exp(log_dens_pred)
    
    # print(p, q)

    # calculate (P || Q)
    kl = kl_div(p, q)
    kl = np.nan_to_num(kl, posinf=0.0)
    kl = np.nanmean(kl)
    
    # calculate JS(P || Q)
    js = jensenshannon(p, q, base=2)    # print (kl)

    ##################################################################
           
            
        # errorTrain = np.mean((gpr.predict(XtrainR.iloc[train_index,:]) - ytrainR.iloc[train_index])**2)
        # errorTest  = np.mean((gpr.predict(XtrainR.iloc[test_index,:]) - ytrainR.iloc[test_index])**2)
        # errorsTrain.append(errorTrain)
        # errorsTest.append(errorTest)
    
    result = [kernel.get_params()['k1__k2__length_scale'], 
              kernel.get_params()['k2__noise_level'], 
              np.mean(scoresTrain), 
              np.mean(errorsTrain), 
              np.mean(errorsTest),
              kl, js
             ]
    
    results = results + [result]
    
results = np.array(results)
# print (results)

results.shape
# -

x = results[:,0]
# plt.plot(result[:,0], '.-b', label='x')
plt.plot(results[:,3], '.-r', label='train')
plt.plot(results[:,4], '-g', label='test')
# plt.plot(results[:,4]-result[:,3], '.-r')
plt.plot((results[:,4]+results[:,3])/2, '-b', label='avg')
# plt.plot(results[:,5], '-k', label='kl')
plt.plot(results[:,6], '-k', label='js')
# plt.xlim(80, 100)
# plt.ylim(-0.1, 0.1)
# plt.ylim(1.0, 1.05)
# plt.ylim(1.1, 1.2)
plt.ylabel('mse')
plt.grid(True)
plt.legend()


x = results[:,0]
# plt.plot(result[:,0], '.-b', label='x')
# plt.plot(results[:,3], '-r', label='train')
# plt.plot(results[:,4], '-g', label='test')
plt.plot(results[:,4]-results[:,3], '-r')
# plt.plot((results[:,4]-results[:,3]), '-b', label='avg')
plt.xlim(80, 100)
# plt.ylim(-0.021, 0.021)
# plt.ylim(1.02, 1.03)
plt.ylabel('mse')
plt.grid(True)
# plt.legend()


kernel.get_params()

df = pd.DataFrame(results, columns=['length_scale', 
                                    'noise_level', 
                                    'scoresTrain', 
                                    'errorsTrain', 
                                    'errorsTest', 
                                    'kl', 
                                    'js'] )

df

plt.plot(df['length_scale'])

xxx

# +
from sklearn.model_selection import KFold
import itertools

# parRange = [np.arange(1, 5, 1), np.arange(0.1, 1, 0.1)]
parRange = [np.logspace(0, 1, 5), np.logspace(-1,1,5)]

kernels = [ConstantKernel(s**2, constant_value_bounds="fixed") * \
           RBF(length_scale=a, length_scale_bounds="fixed")+ \
           WhiteKernel(noise_level=b, noise_level_bounds="fixed") \
           for a, b in list(itertools.product(*parRange))]

for kernel in kernels:
    gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0)

    X = XtrainR
    kf = KFold(n_splits=10)

    scoresTrain = []
    scoresTest = []
    for train_index, test_index in kf.split(X):
        gpr.fit(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index])
        scoresTrain.append(gpr.score(XtrainR.iloc[train_index,:], ytrainR.iloc[train_index]))
        scoresTest.append(gpr.score(XtrainR.iloc[test_index,:], ytrainR.iloc[test_index]))

    # print (kernel)
    print (np.mean(scoresTrain), np.mean(scoresTest))

# -
XtrainR.iloc[test_index]

from sklearn.model_selection import KFold

kf = KFold(n_splits=10)

kf.split(XtrainR)

XtrainR

# +
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
scores = []
for train, test in kf.split(XtrainR):
    # print (train)
    # print (test)
    gpr.fit(XtrainR.iloc[train],ytrainR.iloc[train])
    scores.append(gpr.score(XtrainR.iloc[test], ytrainR.iloc[test]))
print (np.mean(scores))
# -

