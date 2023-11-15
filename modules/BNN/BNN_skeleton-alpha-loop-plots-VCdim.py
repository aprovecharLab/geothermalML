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

# +
import numpy as np
import matplotlib.pyplot as plt

import pickle

import pandas as pd

import datetime
import time
# -

# ## read in results file

# +
summary_filename = 'summary_results_test.pkl'

# read python dict back from the file
with open(summary_filename, 'rb') as fin:
    summary_dict = pickle.load(fin)
# -

summary_dict.keys()

# +
nFeatures = summary_dict['nFeatures']

results = summary_dict['results']
scores = summary_dict['scores']
# -

print(results.shape, scores.shape)

# ### code used to write results dictionary

# +
# ###################################################
# result = [weightScale,
#           Loss, nDOF, 
#           entropy_change, BIC, 
#           AIC, AICc, 
#           avgTrainAcc, avgTestAcc]

# ###################################################
# score = [precision_score(y_true, y_pred, average="macro"),
#          recall_score(y_true, y_pred , average="macro"),
#          f1_score(y_true, y_pred , average="macro"),
#          roc_auc_score(y_true, y_pred)]

# -

# ## summary plots of varying alpha

# +
# plt.rc('figure', figsize=(16, 12))

rowcol = np.array([[1,2],
                   [3,4],
                   [5,6],
                   [7,8]])

ytxt = np.array([['data loss', '# DOF'], 
                 [r'$\Delta$ S', 'BIC'], 
                 ['AIC', 'AICc'], 
                 ['train acc', 'test acc']])

rows, cols = 4, 2

fig, axes = plt.subplots(rows,cols, figsize=(16, 12))
for row in range(rows):
    for col in range(cols):
        print(row, col, rowcol[row,col], ytxt[row,col])
        axes[row,col].plot(results[:,0], results[:,rowcol[row,col]], '-o')
        
        axes[row,col].grid(True)
        axes[row,col].set_ylabel(ytxt[row,col], fontsize=16)
        if row==3:
            axes[row,col].set_xlabel(r'$\alpha$', fontsize=20)
                                                 
# plt.savefig('bayes_loop.png')
# -
# ### truncate if last few points are bad

# +
ignore = 2

results = results[:-ignore]

scores = scores[:-ignore]


# +
# model complexity measure: equation 2.16 of Abu-Mostafa

def Omega(N, dVC, delta):
    
    C = np.sqrt((8/N) * np.log(4*((2*N)**dVC + 1)/delta))
    
    return C


# +
# dVC = np.linspace(1,50)
dVC = results[:,2]

# idxKeep = np.where(dVC>10)[0]
# idxKeep = np.where(dVC>(nFeatures-1))[0]
idxKeep = np.where(dVC>(nFeatures-5))[0]

dVC = dVC[idxKeep]

dataLoss = results[idxKeep,1]

N=100000

# complexity = Omega(N, dVC, 0.001)
complexity = Omega(N, dVC, 0.0001)

# +
plt.rc('figure', figsize=(12, 4))

plt.plot(dVC, dataLoss, 'o', label='in sample error')

plt.plot(dVC,complexity, 'or', label='model complexity')
plt.plot(dVC,dataLoss+complexity, 'og', label='out of sample error')

plt.legend(fontsize=12, loc=5)

plt.xlabel('nDOF $\sim d_{VC}$', fontsize=20)
plt.ylabel('error', fontsize=20)

plt.grid(True)
# -

# # try this by first fitting some simple bilinear or decaying curve to nDOF vs loss

# ## fit nonlinear curves

from scipy.optimize import curve_fit


# +
def nonlinear_ascend(x, a, b, c):
#     return a*np.log(b+x)+c
    y = a/(1 + np.exp(-b*(x-c))) # sigmoid function
    return y

def nonlinear_decay(x, a, b, c):
    return a*np.log(b+x)+c
    # y = a/(1 + np.exp(-b*(x-c))) # sigmoid function
    # y = 1/y
    return y


# -

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
def piecewise_linear0(x, x0, y0, k1):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:y0])


x = results[:,0]
y2 = results[:,2] # nDOF
y3 = results[:,3] # dS



# +
popt_pwl2, pcov_pwl2 = curve_fit(piecewise_linear, x, y2, p0=[2, 20, -2, -1])
# popt_nl2, pcov_nl2 = curve_fit(nonlinear_decay, x, y2, p0=[3,-1,1])
popt_nl2, pcov_nl2 = curve_fit(nonlinear_decay, x, y2, p0=[3,0,1])

popt_nl3, pcov_nl3 = curve_fit(nonlinear_ascend, x, y3, p0=[3,-2,1])
popt_pwl3, pcov_pwl3 = curve_fit(piecewise_linear0, x, y3, p0=[3, 4, 1])

# print (popt_nl2)
print (popt_pwl2)
print (popt_pwl3)

# +
popt_nlVC, pcov_nlVC = curve_fit(nonlinear_decay, dVC, dataLoss, p0=[2,-0.1,10])

fitLoss = nonlinear_decay(dVC, *popt_nlVC)

# +
plt.rc('figure', figsize=(12, 4))

plt.plot(dVC, dataLoss, '.b')
plt.plot(dVC, fitLoss, 'ob')

plt.plot(dVC,complexity, 'or', label='model complexity')
plt.plot(dVC,fitLoss+complexity, 'og', label='out of sample error')

plt.grid(True)

# -

# ### characteristic values of alpha from bilinear fits

alpha_nDOF_knee = np.round(popt_pwl2[0],2)
alpha_dS_max = np.round(popt_pwl3[0],2)

# ### find alpha where nDOF = total number of features

# +
from scipy.optimize import fsolve
def f(x):
    y = piecewise_linear(x, *popt_pwl2)
    return y-nFeatures

starting_guess = np.ceil(popt_pwl2[0])

alpha_nDOF_nFeatures = np.round(fsolve(f, starting_guess),2)[0]
# -

# ### best values of alpha

print ('alpha nDOF_knee: ', alpha_nDOF_knee)
print ('alpha dS_max: ', alpha_dS_max)
print ('alpha nDOF=nFeatures: ', alpha_nDOF_nFeatures)

# ## summary plots of fits used to determine alpha

# +
# %matplotlib inline
# # %matplotlib notebook

plt.rc('figure', figsize=(16, 4))
plt.rc('axes', linewidth=2)

plt.subplot(121)

plt.scatter(x, y2, label="Data")
plt.plot(x, piecewise_linear(x, *popt_pwl2), 'b-', label="piecewise linear")

plt.plot(alpha_nDOF_knee, piecewise_linear(alpha_nDOF_knee, *popt_pwl2), 'or',
         markersize=10, label='knee')
plt.plot(alpha_nDOF_nFeatures, piecewise_linear(alpha_nDOF_nFeatures, *popt_pwl2), 'og', 
         markersize=10, label=' # Features='+str(nFeatures))
# plt.plot(alpha_dS_max, piecewise_linear(alpha_dS_max, *popt_pwl2), '^r',
#          markersize=10)
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel('# DOF', fontsize=20)

plt.grid(True, linewidth=1.5)

plt.legend(fontsize=14, framealpha=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#######################################################3
plt.subplot(122)

plt.scatter(x, y3, label="Data")
plt.plot(x, piecewise_linear0(x, *popt_pwl3), 'b-', label="piecewise linear")

# plt.plot(alpha_nDOF_knee, piecewise_linear0(alpha_nDOF_knee, *popt_pwl3), 'or',
#          markersize=10)
plt.plot(alpha_nDOF_nFeatures, piecewise_linear0(alpha_nDOF_nFeatures, *popt_pwl3), 'og',
         markersize=10, label=' # Features='+str(nFeatures))
plt.plot(alpha_dS_max, piecewise_linear0(alpha_dS_max, *popt_pwl3), 'or',
         markersize=10, label='knee')
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'$\Delta$ S', fontsize=20)

plt.grid(True, linewidth=1.5)

plt.legend(fontsize=14, framealpha=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax = plt.gca()
ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)

plt.tight_layout()

# plt.savefig('BNN_alpha_loop_fits_300dpi.png', dpi=300)
# plt.savefig('BNN_alpha_loop_fits_600dpi.png', dpi=600)
# -

# ## classification report plots

# +
# %matplotlib inline

plt.rc('figure', figsize=(16, 6))

rowcol = np.array([[1,2],
                   [3,4]])

ytxt = np.array([['precision_score', 'recall_score'], 
                 ['f1_score', 'roc_auc_score']])

rows, cols = 2, 2

fig, axes = plt.subplots(rows,cols)
for row in range(rows):
    for col in range(cols):
        print (rowcol[row,col])
        axes[row,col].plot(scores[:,0], scores[:,rowcol[row,col]], '-o')
        
        axes[row,col].grid(True)
        axes[row,col].set_ylabel(ytxt[row,col], fontsize=16)
        if row==1:
            axes[row,col].set_xlabel(r'$\alpha$', fontsize=20)


# -




