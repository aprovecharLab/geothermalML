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

# # Skeleton of an Variational Bayes Artificial Neural Network (BNN) in PyTorch applied to Nevada PFA Geothermal Resources Dataset

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

from torch.utils.data import WeightedRandomSampler

# from torchvision import models
from torchsummary import summary

import datetime
import time

from pathlib import Path
import sys

import h5py

from tqdm.notebook import trange, tqdm

# %matplotlib inline
# -

# # Training Setup

# ### global variables

# +
# ALPHA = 3
# ALPHA = 4
# ALPHA = 0.1
# -


BATCH_SIZE = 64

TRAIN_EPOCHS = 1000



# ## BEGIN data preprocessing

# ## load preprocessed data

# +
path = '../../datasets/'

filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

hf5File = path+filename


# +
f = h5py.File(hf5File, 'r')

def keys(f):
    return [key for key in f.keys()]

key_list = keys(f)
print(key_list)

f.close()
# -

dfXAll = pd.read_hdf(hf5File, key='X')
dfyAll = pd.read_hdf(hf5File, key='y')
XyInfo = pd.read_hdf(hf5File, key='XyInfo')

print( dfXAll.shape, dfyAll.shape, XyInfo.shape)

dfXAll.head()

dfyAll.head()

XyInfo.head()

columns=dfXAll.columns

# ### balance the dataset?

# balance = None
# balance = 'truncate'
balance = 'weighted'

# ### balance dataset method 1: by truncation

if balance == 'truncate':
    dfyAll_neg = dfyAll[dfyAll==0]
    dfyAll_pos = dfyAll[dfyAll==1]
    dfXAll_neg = dfXAll.loc[dfyAll_neg.index]
    dfXAll_pos = dfXAll.loc[dfyAll_pos.index]
    
    # find out which set is smaller
    imin = np.argmin([len(dfyAll_neg), len(dfyAll_pos)])
    
    if imin==0:
        dfyAll_pos = dfyAll_pos.sample(n=len(dfyAll_neg))
        dfXAll_pos = dfXAll_pos.loc[dfyAll_pos.index]
    elif imin==1:
        dfyAll_neg = dfyAll_neg.sample(n=len(dfyAll_pos))
        dfXAll_neg = dfXAll_neg.loc[dfyAll_neg.index]

    dfXAll = pd.concat([dfXAll_neg, dfXAll_pos])
    dfyAll = pd.concat([dfyAll_neg, dfyAll_pos])
    XyInfo = XyInfo.loc[dfXAll.index]


# ### balance dataset method 2: design weighted sampler for dataloaders

if balance == 'weighted':
    class_counts = dfyAll.value_counts()
    class_weights = class_counts/len(dfyAll)

    print(class_counts.values,  class_weights.values)

# ### break into features and labels

XAll = dfXAll
yAll = dfyAll

XAll.shape

yAll.shape

XAll.columns

yAll.name

# ### train/test split

X_trainAll, X_testAll, y_trainAll, y_testAll = train_test_split(
    XAll, yAll, test_size=0.33, random_state=42)
    # XAll, yAll, test_size=0.33)


print (X_trainAll.shape, y_trainAll.shape, X_testAll.shape, y_testAll.shape)

X_trainAll.head()

y_trainAll.head()

print(y_trainAll.shape, y_trainAll.sum())

columns = X_trainAll.columns.to_list()
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
# featureSets

# +
feature_set = 0

columns = featureSets[feature_set]

columns
# -

# ## END data preprocessing



# ## fix input types and dimensions

# +
X_train = X_trainAll[columns].copy()
X_test = X_testAll[columns].copy()

y_train = y_trainAll.copy()
y_test = y_testAll.copy()
# -

nFeatures = X_train.shape[1]
nFeatures

print (X_train.shape, y_train.shape)

# ### balance dataset method 2: using weighted samplers in the dataloaders

if balance == 'weighted':

    # training dataloader sampler
    sample_weights = [1-class_weights[i] for i in y_train.to_numpy()]
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(y_train), replacement=True)

    # testing dataloader sampler
    sample_weights_test = [1-class_weights[i] for i in y_test.to_numpy()]
    sampler_test = WeightedRandomSampler(weights=sample_weights_test, 
                                         num_samples=len(y_test), replacement=True)

# # Build model

# ### use GPU

# +
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
print('Using device:', DEVICE)
print()

#Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
# -

# ## custom dataset class to put random numbers in certain columns each call

# +
# class CustomDataset(Dataset):
#     def __init__(self, X, y, randomList=[]):
#         self.X = X.clone() # without clone this seems to retain changes in original array
#         self.y = y.clone()
#         self.randomList = randomList
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         if self.randomList:
#             self.X[idx,self.randomList] = torch.rand(len(self.randomList))*0
#         return self.X[idx], self.y[idx]

# +
# # test new custom datasets

# features = Tensor(X_train.to_numpy())
# labels = Tensor(y_train.to_numpy()).long()

# dataset = CustomDataset(features, labels, randomList=[9,10,11,12,13,14])
# # dataset = CustomDataset(features, labels)
# print('Number of samples in the data: ', len(dataset))
# print(dataset[2])
# print(dataset[0:5])

# +
# feature columns to continually randomize

# randomList = [9,10,11,12,13,14]
# -

# ## create datasets and batch loaders

# +
###################################################################
# Create train dataset from several tensors with matching first dimension
features = Tensor(X_train.to_numpy())
labels = Tensor(y_train.to_numpy()).long()

train_dataset = TensorDataset( features, labels )
# train_dataset = CustomDataset( features, labels, randomList ) # randomize certain columns

# Create a data loader from the dataset
if balance == None or balance == 'truncate':
    # Type of sampling and batch size are specified at this step
    train_loader = DataLoader(train_dataset, shuffle=True, 
                              batch_size=BATCH_SIZE, drop_last=True)

if balance == 'weighted':
    # weighted sampler to balance training data
    train_loader = DataLoader(train_dataset, sampler=sampler, 
                              batch_size=BATCH_SIZE, drop_last=True)

###################################################################
# Create test dataset from several tensors with matching first dimension
features = Tensor(X_test.to_numpy())
labels = Tensor(y_test.to_numpy()).long()

test_dataset = TensorDataset( features, labels )
# test_dataset = CustomDataset( features, labels, randomList ) # randomize certain columns

# Create a data loader from the dataset
if balance == None or balance == 'truncate':
    test_loader = DataLoader(test_dataset, shuffle=False, 
                             batch_size=BATCH_SIZE, drop_last=True)

if balance == 'weighted':
    # need this weighted sampler to balance testing data too, 
    #      otherwise learning curves are odd
    # a weighted test_loader is a good substitute for the required weighting 
    #      of the test accuracy in imbalanced cases ... 
    #      since accuracies are means over epochs statistics are incorrect without it
    #
    # test_loader = DataLoader(test_dataset, shuffle=False, 
    #                          batch_size=batch_size, drop_last=False)
    # test_loader = DataLoader(test_dataset, sampler=sampler_test, 
                             # batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=sampler_test, 
                             batch_size=BATCH_SIZE, drop_last=True)    


# -

# ## vanilla pytorch bayes by backprop

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
#         epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        epsilon = self.normal.sample().to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def entropy(self, input):
#         entropy = 0.5 * torch.log(2 * math.pi * math.e * self.sigma**2)
        entropy = -self.normal.log_prob(input).exp() * self.normal.log_prob(input)
        return entropy.sum()

    def log_prob(self, input):
        return (- math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


# +
from scipy.stats import gennorm

class GenNormal(object):
    def __init__(self, beta, scale):
        super().__init__()
        self.beta = beta.cpu().numpy()
        self.scale = scale.cpu().numpy()
        self.gennormal = gennorm(self.beta, loc=0.0, scale=self.scale)
    
    @property
    def sigma(self):
        return self.gennormal.std()
    
    def sample(self):
        rv = self.gennormal.rvs(self.scale.shape[0])
        rv = torch.FloatTensor(rv).to(DEVICE)
        return rv
    
    def entropy(self, input):
        x = input.cpu().detach().numpy()
        entropy = -self.gennormal.pdf(x) * self.gennormal.logpdf(x)
        entropy = torch.FloatTensor(entropy)
        return entropy.sum()

    def log_prob(self, input):
        x = input.cpu().detach().numpy()
#         x = input
        logprob = self.gennormal.logpdf(x)
        logprob = torch.FloatTensor(logprob)
        return logprob.sum()


# -

class Normal(object):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.normal = torch.distributions.Normal(loc=0.0, scale=self.scale)
        
    def sample(self):
        sample = self.normal.sample().to(DEVICE)
        return sample
    
    def entropy(self):
        entropy = self.normal.entropy()
        return entropy

    def log_prob(self, input):
        prob = self.normal.log_prob(input)
        return prob.sum()


# ## prior starting parameters

# +
BETAw = torch.FloatTensor([10]).to(DEVICE)
BETAb = torch.FloatTensor([10]).to(DEVICE)

SCALEw = torch.FloatTensor([0.3]).to(DEVICE)
SCALEb = torch.FloatTensor([0.3]).to(DEVICE)
# -

# ## view the prior

prior = Normal(SCALEw)
# prior = GenNormal(BETAw, SCALEw)

prior_x = torch.FloatTensor(np.linspace(-10,10,1000))
# print(x)
# x = (np.linspace(-5,5,1000))
prior_pdf = np.array([])
for xsample in prior_x:
#     xsample = xsample.cpu()
    p = np.exp(prior.log_prob(xsample).cpu().numpy())
    prior_pdf = np.append(prior_pdf, p)    

# +
prior_samples=np.array([])
for i in range(10000):
    sample = prior.sample()
    if abs(sample) < 10:
        prior_samples = np.append(prior_samples,sample.cpu().detach().numpy())
    
kurs = kurtosis(prior_samples, fisher=False)

print (kurs)

# +
plt.rc('figure', figsize=(8, 3))

plt.hist(prior_samples, 50, density=True, label='prior');
plt.plot(prior_x, prior_pdf);
plt.grid(True)

# plt.xlim(-0.5,0.5)

plt.title('prior: normal(scale='+str(SCALEw.cpu().numpy()[0])+')');
# plt.title('prior: gennormal(beta=10, scale=0.3');
# plt.savefig('figures/prior-gennormal.png')

# plt.savefig('figures/prior-normal('+str(SCALEw.cpu().numpy()[0])+')'+'.png')

# -

from sklearn import mixture

X = prior_samples.reshape(-1, 1)

# gmm = mixture.GaussianMixture(n_components = 2, max_iter=100000, covariance_type='diag').fit(X)
gmm = mixture.BayesianGaussianMixture(n_components = 2, max_iter=100000).fit(X)

print (gmm.means_.squeeze())
print ("")
print (np.sqrt(gmm.covariances_).squeeze())


# ## Bayesian modules

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu  = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        
        # Bias parameters
        self.bias_mu  = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        
        # Prior distributions
#         self.weight_prior = GenNormal(BETAw, SCALEw)
#         self.bias_prior   = GenNormal(BETAb, SCALEb)

        self.weight_prior = Normal(SCALEw)
        self.bias_prior   = Normal(SCALEb)

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, nFeatures, nHidden, nLabels, complexityWeight):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden   = nHidden
        self.nLabels   = nLabels
        
        self.complexityWeight = complexityWeight
        
        self.l1 = BayesianLinear(self.nFeatures, self.nHidden)
        self.l2 = BayesianLinear(self.nHidden, self.nHidden)
        self.l3 = BayesianLinear(self.nHidden, self.nLabels)
        
        self.LeakyRelu  = nn.LeakyReLU(0.1)
        self.BatchNorm  = nn.BatchNorm1d(self.nHidden)
        self.dropout    = nn.Dropout(p=0.3)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, sample=False):
        x = x.view(-1, self.nFeatures)
        
        x = self.l1(x, sample)
#         x = self.BatchNorm(x)
        x = self.LeakyRelu(x)

        x = self.l2(x, sample)
        x = self.LeakyRelu(x)

        x = self.l3(x, sample)
        x = self.LogSoftmax(x)
        
        return x
    
    def mfvi_forward(self, inputs, stat=None, q=None, sample_nbr=10):
        """
        Perform mean-field variational inference for the variational estimator model:
            Performs sample_nbr forward passes with uncertainty on the weights, 
            returning its mean and standard deviation
        """
        # logits from all samples - are these logsoftmax called logits?
        logits = torch.stack([self(inputs, sample=True) for _ in range(sample_nbr)])
        # convert logits to probability
        probability = torch.exp(logits)
        
        if   stat == 'mean':
            value = torch.mean(probability, dim=0)
        elif stat == 'stddev':
            value = torch.std(probability, dim=0)
        elif stat == 'mode':
            value, _ = torch.mode(probability, dim=0)
        elif stat == 'quantile':
            value = torch.quantile(probability, q, dim=0)
        else:
            value = torch.tensor([0.0])            
        return value

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def get_weights(self, param):
        if param == 'samples':
            return self.l1.weight.sample(), self.l2.weight.sample(), self.l3.weight.sample()
        elif param == 'sigmas':
            return self.l1.weight.sigma, self.l2.weight.sigma, self.l3.weight.sigma
        elif param == 'means':
            return self.l1.weight.mu, self.l2.weight.mu, self.l3.weight.mu
    
    def get_biases(self, param):
        if param == 'samples':
            return self.l1.bias.sample(), self.l2.bias.sample(), self.l3.bias.sample()
        elif param == 'sigmas':
            return self.l1.bias.sigma, self.l2.bias.sigma, self.l3.bias.sigma
        elif param == 'means':
            return self.l1.bias.mu, self.l2.bias.mu, self.l3.bias.mu

    def sample_elbo(self, input, target, samples=3):
        outputs = torch.zeros(samples, BATCH_SIZE, self.nLabels).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        
        for i in range(samples):
            outputs[i] = self(input, sample=True) # this is same as "forward"
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()

        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction='mean')

        loss = self.complexityWeight*(log_variational_posterior - log_prior) + negative_log_likelihood

        return loss, log_prior, log_variational_posterior, negative_log_likelihood


def train(net, optimizer, epoch):
    
    ###########
    net.train()
    total = 0
    total_loss = 0.0
    total_log_prior = 0.0
    total_log_variational_posterior = 0.0
    total_negative_log_likelihood = 0.0
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(features, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        total_loss += loss.cpu().detach().numpy()
        total_log_prior += log_prior.cpu().detach().numpy()
        total_log_variational_posterior += log_variational_posterior.cpu().detach().numpy()
        total_negative_log_likelihood += negative_log_likelihood.cpu().detach().numpy()
        
    epoch_loss = total_loss / total
    epoch_log_prior = total_log_prior / total
    epoch_log_variational_posterior = total_log_variational_posterior / total
    epoch_negative_log_likelihood = total_negative_log_likelihood / total

    ###########
    net.eval()
    # average train accuracy per epoch ... since this is an average they should be weighted
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            features, labels = data
            outputs = net(features.to(DEVICE), sample=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_train_acc = 100 * correct / total
    
    # average test accuracy per epoch ... since this is an average they should be weighted
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            outputs = net(features.to(DEVICE), sample=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_test_acc = 100 * correct / total

    return epoch_loss, epoch_train_acc, epoch_test_acc, \
            epoch_log_prior, epoch_log_variational_posterior, epoch_negative_log_likelihood


def evaluate(net, epoch):
    
    ###########
    net.eval()
    total = 0
    total_loss = 0.0
    total_log_prior = 0.0
    total_log_variational_posterior = 0.0
    total_negative_log_likelihood = 0.0
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(DEVICE), labels.to(DEVICE)
#         net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(features, labels)
#         loss.backward()
#         optimizer.step()
        total += labels.size(0)
        total_loss += loss.cpu().detach().numpy()
        total_log_prior += log_prior.cpu().detach().numpy()
        total_log_variational_posterior += log_variational_posterior.cpu().detach().numpy()
        total_negative_log_likelihood += negative_log_likelihood.cpu().detach().numpy()
        
    epoch_loss = total_loss / total
    epoch_log_prior = total_log_prior / total
    epoch_log_variational_posterior = total_log_variational_posterior / total
    epoch_negative_log_likelihood = total_negative_log_likelihood / total

    ###########
    net.eval()
    # average train accuracy per epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            features, labels = data
            outputs = net(features.to(DEVICE), sample=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_train_acc = 100 * correct / total
    
    # average test accuracy per epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            outputs = net(features.to(DEVICE), sample=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_test_acc = 100 * correct / total

    return epoch_loss, epoch_train_acc, epoch_test_acc, \
            epoch_log_prior, epoch_log_variational_posterior, epoch_negative_log_likelihood


# ## model instance

# ### settings

# +
TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)

# total number of probabilistic weights + biases
nHidden = 16
nLabels = 2
nBayesianModules = nFeatures*nHidden + \
                     nHidden*nHidden + \
                     nHidden*nLabels + \
                     nHidden + nHidden + nLabels

weightScales = np.around(np.logspace(np.log10(1),np.log10(5),30),3)
# weightScales = np.around(np.logspace(np.log10(1),np.log10(5),10), 3)
# weightScales=[3]
# -

weightScales

len(weightScales)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def to_class(ypred,threshold):
    yclass = np.zeros(ypred.shape,dtype='int')
    yclass[ypred > threshold] = 1
    return yclass


# ### main loop over weightScales

# +
results = []
scores = []
confusions = []

for weightScale in weightScales:
    
    try:
        del net
        del optimizer
        del scheduler
    except:
        pass

    complexityWeight = weightScale/nBayesianModules
#     print ('weight scale = ',weightScale)

    ###################################################
    # instantiate
    net = BayesianNetwork(nFeatures, nHidden, nLabels, complexityWeight).to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, verbose=True)

    ###################################################
    # train
    
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_acc = []

    epoch_logprior = []
    epoch_logvarprior = []
    epoch_negloglikelihood = []

    for epoch in tqdm(range(TRAIN_EPOCHS)):
        eloss, etrain_acc, etest_acc, elogprior, elogvarprior, enegloglikelihood \
            = train(net, optimizer, epoch)

    # check to see how sampling of trained network looks
    #     eloss, etrain_acc, etest_acc, elogprior, elogvarprior, enegloglikelihood \
    #         = evaluate(net, epoch)

        # adjust learning rate
        # scheduler.step()
        scheduler.step(eloss)

        epoch_train_loss.append(eloss)
        epoch_train_acc.append(etrain_acc)
        epoch_test_acc.append(etest_acc)

        epoch_logprior.append(elogprior)
        epoch_logvarprior.append(elogvarprior)
        epoch_negloglikelihood.append(enegloglikelihood)            

    train_loss = np.asarray(epoch_train_loss)
    train_acc = np.asarray(epoch_train_acc)
    test_acc = np.asarray(epoch_test_acc)

    train_logprior = np.asarray(epoch_logprior)
    train_logvarprior = np.asarray(epoch_logvarprior)
    train_negloglikelihood = np.asarray(epoch_negloglikelihood)


    ###################################################
    # get weight and biases parameters

    param = 'means'
    weightsM = np.hstack([net.get_weights(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_weights(param)[1].cpu().detach().numpy().flatten(),
                         net.get_weights(param)[2].cpu().detach().numpy().flatten()])
    biasesM  = np.hstack([net.get_biases(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_biases(param)[1].cpu().detach().numpy().flatten(),
                         net.get_biases(param)[2].cpu().detach().numpy().flatten()])

    param = 'sigmas'
    weightsS = np.hstack([net.get_weights(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_weights(param)[1].cpu().detach().numpy().flatten(),
                         net.get_weights(param)[2].cpu().detach().numpy().flatten()])
    biasesS  = np.hstack([net.get_biases(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_biases(param)[1].cpu().detach().numpy().flatten(),
                         net.get_biases(param)[2].cpu().detach().numpy().flatten()])

    param = 'samples'
    weightSamples = np.array([])
    for i in range(1000):
        sample = np.hstack([net.get_weights(param)[0].cpu().detach().numpy().flatten(), 
                    net.get_weights(param)[1].cpu().detach().numpy().flatten(),
                    net.get_weights(param)[2].cpu().detach().numpy().flatten()])
        weightSamples = np.append(weightSamples,sample)
        
    print (weightSamples.shape)

    ###################################################
    # derive number of degrees of freedom from S2N of weights
    
    weightsS2N = np.abs(weightsM/weightsS)
    weightsS2N = np.sort(weightsS2N)

    idx = np.where(weightsS2N >= 1)[0]
    nDOF = len(idx)

    ###################################################
    # fit probability for entropy estimate
    X = weightsM.copy().reshape(-1, 1)

    ### gennorm
    params = gennorm.fit(X)

    final_differential_entropy = gennorm.entropy(params[0],loc=params[1],scale=params[2])
    initial_differential_entropy = prior.entropy().cpu().detach().numpy()[0]
    entropy_change = initial_differential_entropy - final_differential_entropy

    ###################################################
    # AIC & BIC
    Loss = np.array([])
    for i in range(1024):
        with torch.no_grad():
            features = Tensor(X_train.to_numpy())
            labels = Tensor(y_train.to_numpy()).long()
            outputs = net(features.to(DEVICE), sample=True)
            loss = torch.nn.NLLLoss()(outputs, labels.to(DEVICE))
            Loss = np.append(Loss, loss.cpu().detach().numpy())
    Loss = Loss.mean()
    
    AIC = 2*Loss + 2*nDOF
    BIC = 2*Loss + np.log(TRAIN_SIZE)*nDOF
    
    AICc = AIC + (2*nDOF*(nDOF+1))/(TRAIN_SIZE-nDOF-1)
    
    ###################################################
    avgTrainAcc = train_acc[-100:].mean()
    avgTestAcc = test_acc[-100:].mean()
    
    ###################################################
    result = [weightScale,
              Loss, nDOF, 
              entropy_change, BIC, 
              AIC, AICc, 
              avgTrainAcc, avgTestAcc]
    
    ###################################################
    # classification report
    
    X = X_test.to_numpy()
    y = y_test.to_numpy()

    net.eval()
    net.to(DEVICE)
    features = Tensor(X).to(DEVICE)

    with torch.no_grad():
        p = net.mfvi_forward(features, stat='mean', q=None, sample_nbr=1024)
        # m = p[0].squeeze() # means
    
    ypred = p[:,1].cpu()
    
    threshold = 0.5
    yclass = to_class(ypred, threshold)
    
    y_true = y.squeeze()
    y_pred = yclass.squeeze()

    score = [weightScale,
             precision_score(y_true, y_pred, average="macro"),
             recall_score(y_true, y_pred , average="macro"),
             f1_score(y_true, y_pred , average="macro"),
             roc_auc_score(y_true, y_pred)]
    
    confusion = [weightScale]+confusion_matrix(y_true, y_pred).flatten().tolist()

    ###################################################
    results.append(result)
    scores.append(score)
    confusions.append(confusion)

    print ('')

results = np.array(results)
scores = np.array(scores)
confusions = np.array(confusions)


# +
summary_dict = {'nFeatures': nFeatures, 
                'featureNames': columns,
                'results': results, 
                'scores': scores, 
                'confusions': confusions}

summary_filename = 'summary_results_test.pkl'

# write python dict to a file
with open(summary_filename, 'wb') as fout:
    pickle.dump(summary_dict, fout,
                protocol=pickle.HIGHEST_PROTOCOL)

# -

# read python dict back from the file
with open(summary_filename, 'rb') as fin:
    summary_dict2 = pickle.load(fin)

summary_dict2.keys()

summary_dict2['confusions'].shape

# ## plot all model results

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
# ## plot more

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



