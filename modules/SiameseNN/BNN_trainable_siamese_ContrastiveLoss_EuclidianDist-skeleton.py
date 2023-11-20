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

# # Skeleton of a Siamese Variational Bayes Artificial Neural Network (BNN) in PyTorch applied to Nevada PFA Geothermal Resources Dataset

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

import rasterio

import datetime
import time

from pathlib import Path
import sys

import h5py

from tqdm.notebook import trange, tqdm

# %matplotlib inline
# -

# # Training Setup

# ALPHA = 4.19
ALPHA = 2.35


BATCH_SIZE = 64

TRAIN_EPOCHS = 1000



# ## BEGIN data preprocessing

# ### define a geotiff base map from PFA_ML study to use for affine transforms

# +
path = '../../datasets/'

basemapFilename = 'BNN_(+)_q-0.05_avg-8192.tif'

basemapFile = path+basemapFilename
# -

# ## define preprocessed data and trained model

# +
# myHome = str(Path.home())

# +
path = '../../datasets/'
modelPath = './'

filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

modelFilename = 'BNN_model_trial_2.35.torch'

hf5File = path+filename
modelFile = modelPath+modelFilename
# -


# ## load feature data

# careful! don't use 'w' or you will overwrite the file
f = pd.HDFStore(hf5File, 'r')
print (f.keys())
f.close()

dfXAll = pd.read_hdf(hf5File, key='X')
dfyAll = pd.read_hdf(hf5File, key='y')
XyInfo = pd.read_hdf(hf5File, key='XyInfo')
dfInfo = pd.read_hdf(hf5File, key='dfInfo')
dfn = pd.read_hdf(hf5File, key='dfn')
nullIndexes = pd.read_hdf(hf5File, key='nullIndexes')

print( dfXAll.shape, dfyAll.shape, XyInfo.shape, dfn.shape, nullIndexes.shape)

# ## load basemap geotiff and get affine transform

# +
raster = rasterio.open(basemapFile)
baseMap = raster.read(1)

baseMap.shape

# +
# plt.imshow(baseMap)
# -


fwd = raster.transform
print (fwd)

# test
rev = ~fwd
rev*fwd

dfXAll.head()

dfyAll.head()

XyInfo.head()

columns=dfXAll.columns

columns

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

trainingColumns = featureSets[feature_set]

trainingColumns
# -

# ## END data preprocessing

# ## predict over study area

Xall = dfn.copy()

# ## select only those features used for training

Xall = Xall[trainingColumns]

len(Xall)

# +
Xall = Xall.copy().to_numpy()

# Xall
# -

Xall.shape

# ## fix input types and dimensions

# +
X_train = X_trainAll[trainingColumns].copy()
X_test = X_testAll[trainingColumns].copy()

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
# setting DEVICE on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# setting DEVICE on GPU if available, else CPU
# DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.DEVICE('cpu')
print('Using DEVICE:', DEVICE)
print()

#Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# -

# ## datasets

class CustomSiameseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone() # without clone this seems to retain changes in original array
        self.y = y.clone()
        
        # random sample pairs of X's, make new y as same or different
        # simple version for balanced random dataset as input
        index1 = np.arange(len(self.X))
        index2 = index1.copy()
        np.random.shuffle(index2)

        y12 = []

        for i1,i2 in zip(index1, index2):
            self.y1 = self.y[i1]
            self.y2 = self.y[i2]
            
            ##########################
            # c = 1 when they are same
            c = int(self.y1==self.y2)
            ##########################
            # c = 0 when they are same
            # c = int(self.y1!=self.y2)
            ##########################
            
            y12.append(c)

        self.X1 = self.X[index1]
        self.X2 = self.X[index2]
        self.y12 = Tensor(np.array(y12)).long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y12[idx]


# ### test new custom datasets

# +
features = Tensor(X_train.to_numpy())
labels = Tensor(y_train.to_numpy().reshape(-1,1)).long()

dataset = CustomSiameseDataset(features, labels)
# dataset = CustomDataset(features, labels)

print('Number of samples in the data: ', len(dataset))
# print (features[:2])
# print (labels[:2].T)
print (dataset[:][0].shape, dataset[:][1].shape, dataset[:][2].shape)
# print(dataset[0:4][0])
# print(dataset[0:4][1])
# print(dataset[0:4][2])
# print(dataset[0:10])def forward(self, output1, output2, label):
    #     # distance = F.pairwise_distance(output1, output2)
    #     distance = self.pdist(output1, output2)
    #     # distance = 1-self.cos(output1, output2)
    #     loss_contrastive = torch.mean((label)*torch.pow(distance, 2) 
    #                                   + (1-label)*torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

# -

# ## create datasets and batch loaders

# ### use siamese version

# +
###################################################################
# Create train dataset from several tensors with matching first dimension
features = Tensor(X_train.to_numpy())
labels = Tensor(y_train.to_numpy()).long()

# train_dataset = TensorDataset( features, labels )
train_dataset = CustomSiameseDataset( features, labels )
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

# test_dataset = TensorDataset( features, labels )
test_dataset = CustomSiameseDataset( features, labels )
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
    #                          batch_size=BATCH_SIZE, drop_last=False)
    # test_loader = DataLoader(test_dataset, sampler=sampler_test, 
                             # batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=sampler_test, 
                             batch_size=BATCH_SIZE, drop_last=True)    


# -

# ## contrastive loss using euclidian distance

class ContrastiveLossXXX(torch.nn.Module):
    """
    Contrastive loss function.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLossXXX, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x1, x2, y):
        self.check_type_forward((x1, x2, y))

        # euclidian distance
        diff = x1 - x2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)

        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()

        self.margin = margin

    def forward(self, dist, label):
        
        dist = 1-torch.exp(-dist)
        # print (dist)

        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss


# ## Siamese Network

# ### BNN

# ### prior starting parameters

SCALEw = torch.FloatTensor([0.3]).to(DEVICE)
SCALEb = torch.FloatTensor([0.3]).to(DEVICE)


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
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


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

# prior = ScaleMixtureNGaussians(PI, SCALEw, SCALEw2)
# prior = StudentT(DFw, SCALEw)
# prior = Cauchy(SCALEw)
prior = Normal(SCALEw)
# prior = GenNormal(BETAw, SCALEw)

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
        self.weight_prior = Normal(SCALEw)
        self.bias_prior   = Normal(SCALEb)

        self.log_prior = 0
        self.log_variational_posterior = 0
        
        # self.weight_value = self.weight.mu
        # self.bias_value = self.bias.mu
        self.weight_value = []
        self.bias_value = []
        
    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample==True:            
            weight = self.weight.sample()
            bias = self.bias.sample()
        elif sample==False:
            weight = self.weight.mu
            bias = self.bias.mu
        elif sample==None:
            weight = self.weight_value
            bias = self.bias_value
            
        # save values for use in freezing network
        self.weight_value = weight
        self.bias_value = bias
            
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
# +
class BayesianNetwork(nn.Module):
    def __init__(self, nFeatures, nHidden, complexityWeight):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden   = nHidden
        # self.nLabels   = nLabels
        
        self.complexityWeight = complexityWeight
        
        self.l1 = BayesianLinear(self.nFeatures, self.nHidden)
        self.l2 = BayesianLinear(self.nHidden, self.nHidden)
        # self.l3 = BayesianLinear(self.nHidden, self.nLabels)
        
        self.LeakyRelu  = nn.LeakyReLU(0.1)
        self.BatchNorm  = nn.BatchNorm1d(self.nHidden)
        self.dropout    = nn.Dropout(p=0.3)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    
        # self.pdist = nn.PairwiseDistance(p=2)    
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    ###############################
    # forward method for single pass
    def forward_twin(self, x, sample=False):
        
        x = x.view(-1, self.nFeatures)
        
        x = self.l1(x, sample)
        x = self.LeakyRelu(x)

        x = self.l2(x, sample)
        # x = self.dropout(x)
        # x = self.LeakyRelu(x)
        # x = self.Tanh(x)

        # x = self.fc3(x)
        
        return x
    
    ###############################
    # forward method for training
    def forward(self, x1, x2):
        self.pdist = nn.PairwiseDistance(p=2)  
        
        x1 = self.forward_twin(x1, sample=True)
        x2 = self.forward_twin(x2, sample=None)
                
        x12 = self.pdist(x1, x2)
               
        return x12
    
    ###############################
    # forward method for easy inference
    # def forward_infer(self, x1, x2, sample=False):
    def forward_infer(self, x1, x2):
        
        self.pdist = nn.PairwiseDistance(p=2)    

        x1 = self.forward_twin(x1, sample=True)
        x2 = self.forward_twin(x2, sample=None)
        
        # cosine similarity: output=1 when similar, -1 when different ... do I need to scale this?
        # x12 = self.cos(x1, x2)
        # x12 = 1-x12 # distance
        
        # euclidean distance
        # pdist = nn.PairwiseDistance(p=2)
        x12 = self.pdist(x1, x2)
        
        # return similarity
        x12 = torch.exp(-x12)
               
        return x12
     
    def mfvi_forward_infer(self, x1, x2, stat=None, q=None, sample_nbr=10):
        """
        Perform mean-field variational inference for the variational estimator model:
            Performs sample_nbr forward passes with uncertainty on the weights, 
            returning its mean and standard deviation
        """
        # similarities from all samples
        similarity = torch.stack([self.forward_infer(x1, x2)
                                  for _ in range(sample_nbr)])
        
        if   stat == 'mean':
            value = torch.mean(similarity, dim=0)
        elif stat == 'stddev':
            value = torch.std(similarity, dim=0)
        elif stat == 'mode':
            value, _ = torch.mode(similarity, dim=0)
        elif stat == 'quantile':
            q = torch.tensor([q]).to(DEVICE)
            value = torch.quantile(similarity, q, dim=0)
        else:
            value = torch.tensor([0.0])            
        return value

#####################################################
# modify below for siamese network

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior
               # + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior
               # + self.l3.log_variational_posterior
    
    def get_weights(self, param):
        if param == 'samples':
            return self.l1.weight.sample(), self.l2.weight.sample() #, self.l3.weight.sample()
        elif param == 'sigmas':
            return self.l1.weight.sigma, self.l2.weight.sigma #, self.l3.weight.sigma
        elif param == 'means':
            return self.l1.weight.mu, self.l2.weight.mu #, self.l3.weight.mu
    
    def get_biases(self, param):
        if param == 'samples':
            return self.l1.bias.sample(), self.l2.bias.sample() #, self.l3.bias.sample()
        elif param == 'sigmas':
            return self.l1.bias.sigma, self.l2.bias.sigma #, self.l3.bias.sigma
        elif param == 'means':
            return self.l1.bias.mu, self.l2.bias.mu #, self.l3.bias.mu

    def sample_elbo(self, input1, input2, target, samples=8):
        # outputs = torch.zeros(samples, BATCH_SIZE, self.nLabels).to(DEVICE)
        outputs = torch.zeros(samples, BATCH_SIZE).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        
        for i in range(samples):
            # outputs[i] = self(input, sample=True) # this is same as "forward"
            outputs[i] = self.forward(input1, input2)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()

        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()

##############
# need new data contrastive loss function here
        # negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction='mean')
#         negative_log_likelihood = F.cross_entropy(outputs.mean(0), target, reduction='mean')
        negative_log_likelihood = criterion(outputs.mean(0), target)

        loss = self.complexityWeight*(log_variational_posterior - log_prior) + negative_log_likelihood

        return loss, log_prior, log_variational_posterior, negative_log_likelihood



# -

def xxaccuracy(outputs, labels):
    correct = 0
    total = 0
    outputs = outputs.cpu()
    labels = labels.cpu()
    for j in range(outputs.size()[0]):
        if ((outputs.data.numpy()[j]>0.5)):
            if labels.data.numpy()[j]==0:
                correct +=1
                total+=1
            else:
                total+=1
        else:
            if labels.data.numpy()[j]==1:
                correct +=1
                total+=1
            else:
                total+=1
    return correct, total


def accuracy(outputs, labels):
    correct = 0
    total = 0
    outputs = outputs.cpu()
    labels = labels.cpu()
    for j in range(outputs.size()[0]):
        total += 1
        if labels.data.numpy()[j]==1:
            # correct += 1-outputs.data.numpy()[j]
            correct += np.exp(-outputs.data.numpy()[j])
        else:
            correct += 1-np.exp(-outputs.data.numpy()[j])

    return correct, total


# # edit below for siamese BNN

# ## begin training code from BNN loop

# ### train

def train(net, optimizer, epoch):
    
    ###########
    net.train()
    total = 0
    total_loss = 0.0
    total_log_prior = 0.0
    total_log_variational_posterior = 0.0
    total_negative_log_likelihood = 0.0
    
    ######################################################################   
    # for batch_idx, (features, labels) in enumerate(train_loader):
    #     features, labels = features.to(DEVICE), labels.to(DEVICE)
    ######################################################################
    for features1, features2, labels in train_loader:

        features1 = features1.to(DEVICE)
        features2 = features2.to(DEVICE) 
        labels = labels.to(DEVICE)
        
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(features1, 
                                                                                              features2, 
                                                                                              labels)
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
    # average train accuracy per epoch
    correct = 0
    total = 0
    with torch.no_grad():
        
    ######################################################################   
        # for data in train_loader:
        #     features, labels = data
        #     outputs = net(features.to(DEVICE), sample=True)
    ######################################################################   
            
        for features1, features2, labels in train_loader:
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE) 
            labels = labels.to(DEVICE)    
            outputs = net.forward(features1, features2)
            
            correct, total = accuracy(outputs,labels)
            
            # _, predicted = torch.max(outputs.data, 0)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    epoch_train_acc = 100 * correct / total
    
    # average test accuracy per epoch
    correct = 0
    total = 0
    with torch.no_grad():
        
    ######################################################################   
        # for data in test_loader:
        #     features, labels = data
        #     outputs = net(features.to(DEVICE), sample=True)
    ######################################################################   
            
        for features1, features2, labels in test_loader:
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE) 
            labels = labels.to(DEVICE)    
            outputs = net.forward(features1, features2)
            
            correct, total = accuracy(outputs,labels)

            # _, predicted = torch.max(outputs.data, 0)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    epoch_test_acc = 100 * correct / total

    return epoch_loss, epoch_train_acc, epoch_test_acc, \
            epoch_log_prior, epoch_log_variational_posterior, epoch_negative_log_likelihood


# ### evaluate

def evaluate(net, epoch):
    
    ###########
    net.eval()
    total = 0
    total_loss = 0.0
    total_log_prior = 0.0
    total_log_variational_posterior = 0.0
    total_negative_log_likelihood = 0.0
    
    ######################################################################   
    # for batch_idx, (features, labels) in enumerate(train_loader):
#         features, labels = features.to(DEVICE), labels.to(DEVICE)
# #         net.zero_grad()
#         loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(features, labels)
# #         loss.backward()
# #         optimizer.step()
    ######################################################################   

    for features1, features2, labels in train_loader:

        features1 = features1.to(DEVICE)
        features2 = features2.to(DEVICE) 
        labels = labels.to(DEVICE)
        
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(features1, 
                                                                                              features2, 
                                                                                              labels)

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

    ######################################################################   
        # for data in train_loader:
        #     features, labels = data
        #     outputs = net(features.to(DEVICE), sample=True)
    ######################################################################   
            
        for features1, features2, labels in train_loader:
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE) 
            labels = labels.to(DEVICE)    
            outputs = net.forward(features1, features2)

            correct, total = accuracy(outputs,labels)
            
            # _, predicted = torch.max(outputs.data, 0)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    epoch_train_acc = 100 * correct / total
    
    # average test accuracy per epoch
    correct = 0
    total = 0
    with torch.no_grad():
        
    ######################################################################   
        # for data in test_loader:
        #     features, labels = data
        #     outputs = net(features.to(DEVICE), sample=True)
    ######################################################################   

        for features1, features2, labels in test_loader:
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE) 
            labels = labels.to(DEVICE)    
            outputs = net.forward(features1, features2)
            
            correct, total = accuracy(outputs,labels)

            # _, predicted = torch.max(outputs.data, 0)
            # total += labels.size(0)
            # correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_test_acc = 100 * correct / total

    return epoch_loss, epoch_train_acc, epoch_test_acc, \
            epoch_log_prior, epoch_log_variational_posterior, epoch_negative_log_likelihood


# ### settings

margin = 0.95

# +
# TRAIN_EPOCHS = 2500
# TRAIN_EPOCHS = 1500

# total number of probabilistic weights + biases
nHidden = 16

# nBayesianModules = nFeatures*nHidden + \
#                      nHidden*nHidden + \
#                      nHidden*nLabels + \
#                      nHidden + nHidden + nLabels
# Siamese network does not have final classifier layer
nBayesianModules = nFeatures*nHidden + \
                     nHidden*nHidden + \
                     nHidden + nHidden

# weightScales = np.linspace(3,5,21)
# weightScales = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, \
#                 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.,  ]
# weightScales = np.around(np.logspace(np.log10(1),np.log10(6),25),3)

# weightScales = [0.01]
# weightScales = [0.075]
weightScales = [0.125]
# weightScales = [0.19]
# weightScales = [0.35]
# weightScales = [0.4]
# -

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

for weightScale in weightScales:

    complexityWeight = weightScale/nBayesianModules
#     print ('weight scale = ',weightScale)

    ###################################################
    # instantiate
    net = BayesianNetwork(nFeatures, nHidden, complexityWeight).to(DEVICE)
    criterion = ContrastiveLoss(margin)

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
                         net.get_weights(param)[1].cpu().detach().numpy().flatten()])
                         # net.get_weights(param)[2].cpu().detach().numpy().flatten()])
    biasesM  = np.hstack([net.get_biases(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_biases(param)[1].cpu().detach().numpy().flatten()])
                         # net.get_biases(param)[2].cpu().detach().numpy().flatten()])

    param = 'sigmas'
    weightsS = np.hstack([net.get_weights(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_weights(param)[1].cpu().detach().numpy().flatten()])
                         # net.get_weights(param)[2].cpu().detach().numpy().flatten()])
    biasesS  = np.hstack([net.get_biases(param)[0].cpu().detach().numpy().flatten(), 
                         net.get_biases(param)[1].cpu().detach().numpy().flatten()])
                         # net.get_biases(param)[2].cpu().detach().numpy().flatten()])

    param = 'samples'
    weightSamples = np.array([])
    for i in range(1000):
        sample = np.hstack([net.get_weights(param)[0].cpu().detach().numpy().flatten(), 
                    net.get_weights(param)[1].cpu().detach().numpy().flatten()])
                    # net.get_weights(param)[2].cpu().detach().numpy().flatten()])
        weightSamples = np.append(weightSamples,sample)

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

#     ###################################################
#     # AIC & BIC
#     L = np.array([])
#     for i in range(1024):
#         with torch.no_grad():
#             features = Tensor(X_train.to_numpy())
#             labels = Tensor(y_train.to_numpy()).long()
#             outputs = net(features.to(DEVICE), sample=True)
#             loss = torch.nn.NLLLoss()(outputs, labels.to(DEVICE))
#             L = np.append(L, loss.cpu().detach().numpy())
#     L = L.mean()
    
#     AIC = 2*L + 2*nDOF
#     BIC = 2*L + np.log(TRAIN_SIZE)*nDOF
    
#     AICc = AIC + (2*nDOF*(nDOF+1))/(TRAIN_SIZE-nDOF-1)
    
#     ###################################################
#     avgTrainAcc = train_acc[-100:].mean()
#     avgTestAcc = test_acc[-100:].mean()
    
    ###################################################
    summary = [weightScale, # L,
               nDOF, entropy_change,] 
               # AIC, AICc, BIC, 
               # avgTrainAcc, avgTestAcc]
    
#     ###################################################
#     # classification report
    
#     X = X_test.to_numpy()
#     y = y_test.to_numpy()

#     net.eval()
#     net.to(DEVICE)
#     features = Tensor(X).to(DEVICE)

#     with torch.no_grad():
#         p = net.mfvi_forward(features, sample_nbr=1024)
#         m = p[0].squeeze() # means
# #         s = p[1].squeeze() # stddevs    
    
#     p_mean = torch.softmax(m.cpu().detach(), dim=1).numpy()
#     ypred = p_mean[:,1]
    
#     threshold = 0.5
#     yclass = to_class(ypred, threshold)
    
#     y_true = y.squeeze()
#     y_pred = yclass.squeeze()

#     report = [precision_score(y_true, y_pred, average="macro"),
#               recall_score(y_true, y_pred , average="macro"),
#               f1_score(y_true, y_pred , average="macro"),
#               confusion_matrix(y_true, y_pred).tolist(),
#               roc_auc_score(y_true, y_pred)]


#     ###################################################
#     results.append([summary, report])
# #     print ([summary, report])
    

    ###################################################
    # save all in pickle file
#     traintest_filename = 'bayesian_traintest_'+ \
#                             'group'+str(grp)+'_'+ \
#                             'featureSet'+str(feature_set)+'_'+ \
#                             'weightScale'+str(weightScale)+ \
#                             '.pkl'

#     traintest_dict = {'group': grp, 'feature_set': feature_set,
#                         'weightScale': weightScale,
#                         'train_loss': train_loss, 
#                         'train_acc': train_acc, 
#                         'test_acc': test_acc,
#                         'train_logprior': train_logprior, 
#                         'train_logvarprior': train_logvarprior, 
#                         'train_negloglikelihood': train_negloglikelihood,
#                         'weightsM': weightsM, 'weightsS': weightsS,
#                         'biasesM': biasesM, 'biasesS': biasesS,
#                         'weightSamples': weightSamples,
#                         'nDOF': nDOF, 
#                         'AIC': AIC, 'AICc': AICc, 'BIC': BIC,
#                         'initial_differential_entropy': initial_differential_entropy,
#                         'final_differential_entropy': final_differential_entropy,
#                         'entropy_change': entropy_change,
#                      }

    # write python dict to a file
#     output = open(traintest_filename, 'wb')
#     pickle.dump(traintest_dict, output)
#     output.close()

    # read python dict back from the file
    # pkl_file = open(traintest_filename, 'rb')
    # traintest_dict2 = pickle.load(pkl_file)
    # pkl_file.close()

    # print (traintest_dict)
    # print (traintest_dict2)

    # del net, criterion, optimizer, scheduler
    
    print ('')
# -

print(summary)

# +
plt.rc('figure', figsize=(16, 4))

plt.subplot(121)
plt.plot(train_loss, label='train')
# plt.plot(test_loss, label='test')
plt.ylabel('loss')
plt.legend()

plt.grid(True)

plt.subplot(122)
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.ylabel('accuracy')
plt.legend()
plt.ylim([0,100])

plt.grid(True)

# plt.savefig('good_training_dropout0.5.png')
# -
# ## end training code from BNN loop



# ## save model

# save model
model_filename = 'BNN_siamese_model_' + \
                    'weightScale_'+str(weightScale)+ \
                    '.torch'
# model_filename = 'finalResults/BNN_model_'+outFileRoot+'.torch'
print (model_filename)
torch.save(net, model_filename)    


# ## begin siamese inference from pretrained BNN model

# ### pretrained model

# model_path = myHome+'/Desktop/PFA_ML_Sandbox_Results/BayesianNN_Models_Maps/savedModels/'
model_path = './'

# trainedModelFilename = 'BNN_model_february2021_tc2_33_SmoothLSS_grp_0_featureSet_0_alpha_3.65.torch'
trainedModelFilename = 'BNN_siamese_model_weightScale_'+str(weightScale)+ \
                    '.torch'
# trainedModelFilename = 'BNN_siamese_model_weightScale_0.35.torch'

modelFile = model_path + trainedModelFilename

# +
# ## load trained model
# # Evaluate model
# ## statistics

# # Load
net = torch.load(modelFile)
net.eval()

# print (net)
# # summary(net, features.shape, col_names=("input_size", "output_size", "num_params"), verbose=2, depth=2);
# -



# ## Inference to make Similarity Map

# ## predict posterior distribution for specific sites in study area

# #### (a) specify sites by rcIndex (['id_rc']), array([[r,c]]), or array([[utmE, utmN]])
# #### (c) specify sites by UTM coordinates

# + active=""
# Gabbs Valley (PFA-discovered, not a labeled training site)
# UTM83: 420033.75, 4301656.14
#
# Granite Springs Valley (PFA-discovered, not a labeled training site)
# UTM83: 337158.24, 4456578.84
#
# McGinness Hills - unless I'm misremembering (existing labeled training site)
# UTM83: 507429.92, 4383670.34
#
# Argena Rise - bright spot
# 40.57095184, -116.77503814
# UTM83: 519172 4491133
# -

# ## logger

# +
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')

# fh = logging.FileHandler('log_report'+timeNow()+'.md')
fh = logging.FileHandler('log_report.md')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)

# logger.info(filename)
# logger.info(note+'\n')
logger.info('begin\n')
# -

# ## specify one case

testIdx = 0

# +
rcArray  = np.array([[0, 0], 
                     [0, 0], 
                     [0, 0], 
                     [0, 0]])

# utmArray = np.array([[420033.75, 4301656.14], 
#                      [337158.24, 4456578.84], 
#                      [507429.92, 4383670.34]])
utmArray = np.array([[420033.75, 4301656.14], 
                     [337158.24, 4456578.84], 
                     [507429.92, 4383670.34],
                     [519172.0, 4491133.0],
                    ])

rcIndex = np.array([])
# rcIndex = ['R1C1']
           
descriptionArray = ([['Gabbs_Valley'],
                    ['Granite_Springs_Valley'],
                    ['McGinness_Hills'],
                    ['Argenta_Rise']])

# utmArray = np.array([])
utmArray = utmArray[testIdx]
# utmArray = np.array([420033.75, 4301656.14])

rcArray = np.array([])
# rcArray = rcArray[0]

siteDescription = descriptionArray[testIdx][0]
# siteDescription = descriptionArray
# -

print (rcIndex, rcArray, utmArray, siteDescription)

logger.info('')
logger.info('```')
logger.info('BNN pretrained Siamese Network with similarity=log(-EuclideanDistance)')
logger.info('site description: '+siteDescription)
logger.info('```')
logger.info('')

# +
if len(rcArray) > 0:
    rcArray.shape = (-1,2)
    
    rcIndex = []
    utmArray = []
    for row, col in rcArray:
        rcString = 'R'+str(1000-row)+'C'+str(col+1)
        rcIndex.append(rcString)
        (UTM_E_NAD83, UTM_N_NAD83) = fwd*(col, row)
        utmArray.append([UTM_E_NAD83, UTM_N_NAD83])
    utmArray = np.asarray(utmArray)

elif len(utmArray) > 0:    
    utmArray.shape = (-1,2)
    
    logger.info('')
    logger.info('```')
    logger.info('site UTM: '+str(utmArray))
    logger.info('```')
    logger.info('')
    
    rcArray = []
    for UTM_E_NAD83, UTM_N_NAD83 in utmArray:
        (c,r) = ~fwd*(UTM_E_NAD83, UTM_N_NAD83)
        rcArray.append([int(r), int(c)])
    rcArray = np.asarray(rcArray)

    rcIndex = []
    utmArray = []
    for row, col in rcArray:
        rcString = 'R'+str(1000-row)+'C'+str(col+1)
        rcIndex.append(rcString)
        (UTM_E_NAD83, UTM_N_NAD83) = fwd*(col, row)
        utmArray.append([UTM_E_NAD83, UTM_N_NAD83])
    utmArray = np.asarray(utmArray)

elif len(rcIndex) > 0:
    
    row = np.array([])
    col = np.array([])
    for s in rcIndex:
        row = np.append(row,int(eval(s.split('R')[1].split('C')[0])))
        col = np.append(col,int(eval(s.split('R')[1].split('C')[1])))
    # row = 999-(row-1).astype('int')
    row = (1000-row).astype('int')

    col = (col-1).astype('int')
    rcArray = np.array([row, col]).T

    utmArray = []
    for row, col in rcArray:
        # rcIndex = 'R'+str(1000-row)+'C'+str(col+1)
        (UTM_E_NAD83, UTM_N_NAD83) = fwd*(col, row)
        utmArray.append([UTM_E_NAD83, UTM_N_NAD83])
    utmArray = np.asarray(utmArray)

print (rcIndex)
print (rcArray)
print (utmArray)
# -

logger.info('')
logger.info('```')
logger.info('pixel rowcol '+str(rcArray))
logger.info('pixel id_rc: '+str(rcIndex))
logger.info('pixel UTM:   '+str(utmArray))
logger.info('```')
logger.info('')

idx = []
for RC in rcIndex:
    i = dfInfo[dfInfo['id_rc'] == RC].index.tolist()[0]
    idx.append(i)
    print (i, RC)
# print (idx)

# ### select features from dataframe

dfXtest = dfn.loc[idx][trainingColumns]

# +
Xtest = torch.Tensor(dfXtest.to_numpy())[0]

Xtest.shape
# -


# ## predict whole study area

# +
# stat = 'quantile'
# quantile = 0.05
# quantile = 0.2

stat = 'mean'

# stat = 'stddev'

N = 512
# N = 128

# +
net.eval()

eval_loader = DataLoader(Tensor(Xall).to(DEVICE), shuffle=False, 
                         batch_size=4096, drop_last=False)

sAll = []

for features in tqdm(eval_loader):
    with torch.no_grad():
        
        if stat == 'quantile':
            statstr = str(quantile)
            s = net.mfvi_forward_infer(Xtest.to(DEVICE), features,
                                       stat='quantile', q=quantile, sample_nbr=N)
        elif stat == 'mean':
            statstr = stat
            s = net.mfvi_forward_infer(Xtest.to(DEVICE), features, 
                                       stat='mean', q=None, sample_nbr=N)
        elif stat == 'stddev':
            statstr = stat
            s = net.mfvi_forward_infer(Xtest.to(DEVICE), features, 
                                       stat='stddev', q=None, sample_nbr=N)
        
        s = s.squeeze()
        sAll.append(s)
        
sAll = torch.cat(sAll, dim=0)

# -

sAll = sAll.cpu().detach().numpy()
sAll.shape

# +
# sAll = np.exp(-sAll)
# -

# ## plot a histogram

# +
plt.rc('figure', figsize=(6,4))

# plt.hist(yall[:,0],50);
plt.hist(sAll,50);
## plt.hist(yall[yall.>0.5],50);
# plt.axis([0,1,0,100000])
plt.grid(True)

plt.title('BNN '+statstr+' Predictions for study area',fontsize=20)
# plt.xlabel('distance from test site',fontsize=18)
plt.xlabel('eSimilarity to test site',fontsize=18)

plt.tight_layout()

figName = 'fig0_'+siteDescription+'_tBNN_siamese_PDF_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
# figName = 'fig0_BNN_'+statstr+'_'+str(testIdx)+'.png'
plt.savefig(figName, dpi=300)
logger.info('')
logger.info('![PDF](./'+figName+' "PDF")')
logger.info('')
# -



# ## make a map

# mask nulls
sAll[nullIndexes] = np.nan

# reshape and flipud
img = np.reshape(sAll,(1000,-1))
img = np.flipud(img)
img.shape

# ### get benchmark sites coordinates, use TrainCode=1

XyInfo.columns

benchmarks = XyInfo[['row', 'column', 'TrainCodePos']].copy()
benchmarks.rename({'TrainCodePos': 'label'}, axis='columns', inplace=True)

benchmarks = benchmarks.loc[(XyInfo.TrainCodePos==1) | (XyInfo.TrainCodeNeg==1)]
benchmarks['label'][benchmarks.label > 1] = 0


benchmarks

# ### fix indexing for matplotlib imshow ... row, col referenced from upper left

indexB = benchmarks.index.astype(int).to_numpy()

benchmarks = benchmarks.astype(int)
benchmarks = benchmarks.to_numpy()

rowB = 999-benchmarks[:,0]
colB = benchmarks[:,1]
labelB = benchmarks[:,2]

rcArray

c = rcArray[0][1]
r = rcArray[0][0]

# ### plot

vmin = 0.6
vmax = 1

# +
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(16,8))

plt.suptitle(siteDescription, fontsize=20)

# im = plt.imshow(baseMap*img, origin='upper', cmap='coolwarm', 
im = plt.imshow(img, origin='upper', cmap='coolwarm',
                norm=colors.Normalize(vmin=vmin, vmax=vmax),
               )

plt.plot(c,r, marker='o', fillstyle='none', 
         # markeredgecolor='orange', markeredgewidth=2, ms=20)
         markeredgecolor='black', markeredgewidth=2, ms=20)

plt.scatter(colB, rowB, c=labelB, s=50, cmap='prism', edgecolor='k')

# plt.title('"new" Fairway - categorical localK', fontsize=18)
# plt.title('ANN model '+str(n_run), fontsize=18)
plt.title('BNN '+statstr+' Siamese Network with BNN training sites', fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

cb = plt.colorbar(im, cax=cax)
# cb.set_label('feature euclidean distance from site', fontsize=16)
cb.set_label('feature eSimilarity to test site', fontsize=16)

plt.tight_layout()

# figName = 'fig1_BNN_'+statstr+'_'+str(testIdx)+'.png'
figName = 'fig1_'+siteDescription+'_tBNN_siamese_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
# figName = 'fig1_BNN_'+statstr+'_lognorm_'+str(testIdx)+'.png'
plt.savefig(figName, dpi=300)
logger.info('')
logger.info('![Map](./'+figName+' "Map")')
logger.info('')

# +
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1,1,figsize=(16,8))

plt.suptitle(siteDescription, fontsize=20)

im0 = ax.imshow(baseMap, origin='upper', cmap='coolwarm', 
                norm=colors.Normalize(vmin=vmin, vmax=vmax),
               )

ax.plot(c,r, marker='o', fillstyle='none', 
         markeredgecolor='black', markeredgewidth=2, ms=20)

ax.set_title('BNN 0.05 Percentile Map', fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

cb = plt.colorbar(im0, cax=cax)
cb.set_label('probability of (+) site', fontsize=16)
# ax[0].scatter(colB, rowB, c=-labelB, s=50, cmap='prism', edgecolor='k')


plt.tight_layout()

# figName = 'fig2_'+siteDescription+'_BNN_siamese_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
figName = 'fig2_'+siteDescription+'_tBNN_percentile_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
# figName = 'fig2_BNN_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
plt.savefig(figName, dpi=300)
logger.info('')
logger.info('![Map](./'+figName+' "Map")')
logger.info('')


# +
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1,1,figsize=(16,8))

plt.suptitle(siteDescription, fontsize=20)

im0 = ax.imshow(img, origin='upper', cmap='coolwarm', 
                norm=colors.Normalize(vmin=vmin, vmax=vmax),
               )

ax.plot(c,r, marker='o', fillstyle='none', 
         markeredgecolor='black', markeredgewidth=2, ms=20)

ax.set_title('BNN '+statstr+' Siamese Network', fontsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)

cb = plt.colorbar(im0, cax=cax)
cb.set_label('feature eSimilarity to test site',fontsize=16)

plt.tight_layout()

figName = 'fig3_'+siteDescription+'_tBNN_siamese_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.png'
plt.savefig(figName, dpi=300)

logger.info('')
logger.info('![Map](./'+figName+' "Map")')
logger.info('')
# -


# ## save maps as geotiffs


import osgeo.gdal as gdal
import osgeo.osr as osr

transform = [286202.6902, 250.0, 0.0, 4508588.7659, 0.0, -250.0]

img0 = img.copy()
# img0 = np.flipud(img)
# img1 = np.flipud(imgm)
# img2 = np.flipud(imgmax)

height = img0.shape[0]
width = img0.shape[1]
print (width, height)

# +
# open a new file
# geotiffFile = 'argenta_pretrained_BNN_siamese'+'.tif'
geotiffFile = siteDescription+'_tBNN_siamese_'+statstr+'_'+str(testIdx)+'_avg_'+str(N)+'.tif'

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



