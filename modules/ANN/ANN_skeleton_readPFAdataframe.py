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

# + [markdown] editable=true slideshow={"slide_type": ""} jp-MarkdownHeadingCollapsed=true
# # Geothermal and Machine Learning Sandbox
# -

# # Skeleton of an Artificial Neural Network (ANN) in PyTorch applied to Nevada PFA Geothermal Resources Dataset

# + editable=true slideshow={"slide_type": ""}
import numpy as np
import matplotlib.pyplot as plt

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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
# from torchsummary import summary

import datetime
import time

from pathlib import Path
import sys

import h5py

from tqdm.notebook import trange, tqdm

# %matplotlib inline

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## BEGIN data preprocessing
# -

# ## load preprocessed data

# +
path = '../../datasets/'

filename = 'PFA_features&labels_march2021_tc2_imbalanced_SmoothLSS_FSS_df.h5'

hf5File = path+filename


# + editable=true slideshow={"slide_type": ""}
f = h5py.File(hf5File, 'r')

def keys(f):
    return [key for key in f.keys()]

key_list = keys(f)
print(key_list)

f.close()

# + editable=true slideshow={"slide_type": ""}
dfXAll = pd.read_hdf(hf5File, key='X')
dfyAll = pd.read_hdf(hf5File, key='y')
XyInfo = pd.read_hdf(hf5File, key='XyInfo')
# -

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

# + editable=true slideshow={"slide_type": ""}

# -

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

X_train

y_train

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

# batch_size = 1024
batch_size = 64

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
                              batch_size=batch_size, drop_last=False)

if balance == 'weighted':
    # weighted sampler to balance training data
    train_loader = DataLoader(train_dataset, sampler=sampler, 
                              batch_size=batch_size, drop_last=False)

###################################################################
# Create test dataset from several tensors with matching first dimension
features = Tensor(X_test.to_numpy())
labels = Tensor(y_test.to_numpy()).long()

test_dataset = TensorDataset( features, labels )
# test_dataset = CustomDataset( features, labels, randomList ) # randomize certain columns

# Create a data loader from the dataset
if balance == None or balance == 'truncate':
    test_loader = DataLoader(test_dataset, shuffle=False, 
                             batch_size=batch_size, drop_last=False)

if balance == 'weighted':
    # need this weighted sampler to balance testing data too, 
    #      otherwise learning curves are odd
    # a weighted test_loader is a good substitute for the required weighting 
    #      of the test accuracy in imbalanced cases ... 
    #      since accuracies are means over epochs statistics are incorrect without it
    #
    # test_loader = DataLoader(test_dataset, shuffle=False, 
    #                          batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=sampler_test, 
                             batch_size=batch_size, drop_last=True)
# -

# ## FCNN in pytorch

DropoutValue = 0.2
WeightDecayValue = 5e-02


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
        self.dropout = nn.Dropout(p=DropoutValue)
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

# ## clean up from multiple trials

try:
    del net
    del optimizer
    del criterion
except:
    pass

# ## instantiate model

# +
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FCNN(nFeatures, 16, 2).to(DEVICE)
# net = FCNN(nFeatures, 8, 2).to(DEVICE)

#####################################
# initialize weights however you like
#####################################
# torch.nn.init.xavier_uniform_(net.fc1.weight)
# torch.nn.init.zeros_(net.fc1.bias)

# torch.nn.init.xavier_uniform_(net.fc2.weight)
# torch.nn.init.zeros_(net.fc2.bias)

optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=WeightDecayValue)

# criterion = torch.nn.CrossEntropyLoss()
# use negative log likelihood instead since we set LogSoftmax as network output
criterion = torch.nn.NLLLoss() 
# -

# ## train model

# +
startTime = time.time()

epoch_train_loss = []
epoch_train_acc = []
epoch_test_acc = []
        
for epoch in trange(1000):
    
    total = 0
    total_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
        
        y_pred = net.forward(features.to(DEVICE))
        loss = criterion(y_pred, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += labels.size(0)
        total_loss += loss.cpu().detach().numpy()
        
    epoch_train_loss.append(total_loss/total)
        
    # average test accuracy per epoch ... since this is an average they should be weighted
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            outputs = net.forward(features.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_test_acc.append(100 * correct / total)

    # average train accuracy per epoch ... since this is an average they should be weighted
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            features, labels = data
            outputs = net.forward(features.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    epoch_train_acc.append(100 * correct / total)
    
train_loss = np.asarray(epoch_train_loss)
train_acc = np.asarray(epoch_train_acc)
test_acc = np.asarray(epoch_test_acc)

trainTime = time.time() - startTime

print ('Training Time = ', trainTime)
# -


# ## a look at last batch results

# +
plt.rc('figure', figsize=(6, 3))

plt.hist(torch.exp(outputs[:,0]).cpu(), 50, label='class 0');
plt.hist(torch.exp(outputs[:,1]).cpu(), 50, alpha=0.5, label='class 1');
plt.legend()

plt.xlabel('probability of class')
# -

# ## learning curves

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
# plt.ylim([50,100])

plt.grid(True)

# plt.savefig('learning_curves.png')
# +
# xxx
# -

# ## save model

model_filename = 'ANN_model_trial'+ \
                '_dropout-' + str(DropoutValue) + \
                '_weight_decay-'+str(WeightDecayValue) + \
                '.torch'
print (model_filename)
torch.save(net, model_filename)

# ## save learning curve data

# +
# traintest_filename = 'ann_traintest_'+ \
#                     'featureSet'+str(feature_set)+'.pkl'

# +
# traintest_dict = {'group': grp, 'feature_set': feature_set, 
#                   'train_loss': train_loss, 
#                   'train_acc': train_acc, 
#                   'test_acc': test_acc}

# # write python dict to a file
# output = open(traintest_filename, 'wb')
# pickle.dump(traintest_dict, output)
# output.close()

# # read python dict back from the file
# pkl_file = open(traintest_filename, 'rb')
# traintest_dict2 = pickle.load(pkl_file)
# pkl_file.close()

# print (traintest_dict)
# print (traintest_dict2)
# -


xxx



# # Extra Stuff



# ## evaluate the classifier

# +
X = X_test.to_numpy()
y = y_test.to_numpy()

# X = np.vstack((X_train.to_numpy(), X_test.to_numpy()))
# y = np.hstack((y_train.to_numpy(), y_test.to_numpy()))


# +
net.eval()

net.to(DEVICE)
features = Tensor(X).to(DEVICE)

with torch.no_grad():
    p = net.forward(features)
    p = torch.exp(p)
# -

ypred = p[:,1].cpu()

# +
plt.rc('figure', figsize=(12, 3))

plt.hist(ypred,50, label='(+)');
plt.legend()

plt.grid(True)


# +
def to_class(ypred,threshold):
    yclass = np.zeros(ypred.shape,dtype='int')
    yclass[ypred > threshold] = 1
    return yclass

def CrossEntropyLoss(ypred, y):
    if y == 1:
        loss = -np.log(ypred)
    else:
        loss = -np.log(1 - ypred)
    return loss


# +
threshold = 0.5

yclass = to_class(ypred, threshold)
pred_positives = np.sum(yclass)

d = y-yclass
true_positives = np.sum(y)
false_positives = np.sum(np.abs(d[d==-1]))
false_negatives = np.sum(d[d==1])

accuracy = float(pred_positives)/float(true_positives)
precision = true_positives/(true_positives + false_positives)
recall = true_positives/(true_positives + false_negatives)
print (pred_positives, true_positives, false_positives, false_negatives)
print (accuracy, precision, recall)



# +
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

y_true = y.squeeze()
y_pred = yclass.squeeze() 
# labels = ['noise', 'event']

# print (y_true)

# # # Print f1, precision, and recall scores
# print(precision_score(y_true, y_pred, average="macro"))
# print(recall_score(y_true, y_pred , average="macro"))
# print(f1_score(y_true, y_pred , average="macro"))
print('Report for Trained Classifier')
print(classification_report(y_true, y_pred))

print('')
print('Confusion Matrix')
print('[true negative  false positive]')
print('[false negative  true positive]')
print('')
print(confusion_matrix(y_true, y_pred))
print('')


# -


print(precision_score(y_true, y_pred, average="macro"))
print(recall_score(y_true, y_pred , average="macro"))
print(f1_score(y_true, y_pred , average="macro"))
# print(classification_report(y_true, y_pred))

ypred.shape

# +
# roc curve and auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, ypred)

roc_auc = auc(fpr, tpr)

# +
plt.rc('figure', figsize=(6,6))

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

# idx4 = np.where(np.abs(thresholds)>=0.4)[0][-1]
idx5 = np.where(np.abs(thresholds)>=0.5)[0][-1]
idx6 = np.where(np.abs(thresholds)>=0.6)[0][-1]
idx7 = np.where(np.abs(thresholds)>=0.7)[0][-1]
idx8 = np.where(np.abs(thresholds)>=0.8)[0][-1]
idx9 = np.where(np.abs(thresholds)<=0.9)[0][-1]

# plt.plot(fpr[idx4],tpr[idx4],'o', label='threshold=0.4')
plt.plot(fpr[idx5],tpr[idx5],'o', label='threshold=0.5')
plt.plot(fpr[idx6],tpr[idx6],'o', label='threshold=0.6')
plt.plot(fpr[idx7],tpr[idx7],'o', label='threshold=0.7')
plt.plot(fpr[idx8],tpr[idx8],'o', label='threshold=0.8')
plt.plot(fpr[idx9],tpr[idx9],'o', label='threshold=0.9')

plt.legend(loc="lower right")

plt.grid(True)

# plt.savefig('figures/roc.png')
# -

# “A receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied. It is created by plotting the fraction of true positives out of the positives (TPR = true positive rate) vs. the fraction of false positives out of the negatives (FPR = false positive rate), at various threshold settings. TPR is also known as sensitivity, and FPR is one minus the specificity or true negative rate.”

# +
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y, ypred)

# +
plt.rc('figure', figsize=(6,6))

lw = 2
plt.plot(recall, precision, color='darkorange',
         lw=lw, label='PR curve')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision / Recall Curve')

idx5 = np.where(np.abs(thresholds)<=0.5)[0][-1]
idx6 = np.where(np.abs(thresholds)<=0.6)[0][-1]
idx7 = np.where(np.abs(thresholds)<=0.7)[0][-1]
idx8 = np.where(np.abs(thresholds)<=0.8)[0][-1]
idx9 = np.where(np.abs(thresholds)<=0.9)[0][-1]

plt.plot(recall[idx5],precision[idx5],'o', label='threshold=0.5')
plt.plot(recall[idx6],precision[idx6],'o', label='threshold=0.6')
plt.plot(recall[idx7],precision[idx7],'o', label='threshold=0.7')
plt.plot(recall[idx8],precision[idx8],'o', label='threshold=0.8')
plt.plot(recall[idx9],precision[idx9],'o', label='threshold=0.9')

plt.legend(loc="lower center")

plt.grid(True)

# -


# ## Global Sensitivity Analysis

# ### in Sobel and Delta methods can choose to use either log probability (logsoftmax) or probability exp(logsoftmax) as output to look at. Integrated Gradients uses output of network which is log probability

useOutput = 'probability'
# useOutput = 'logprob'

# ## Sobol analysis

# +
import SALib as salib
# from SALib.sample import sobol
from SALib.sample import saltelli
from SALib.analyze import sobol

from SALib.plotting.bar import plot as barplot

# +
Xtest = torch.Tensor(X_test.to_numpy())
# Xtest = torch.Tensor(X_train.to_numpy())
ytest = torch.Tensor(y_test.to_numpy())

ytest[:20]
# -


preds = net.forward(Xtest.to(DEVICE)).cpu().detach()
preds = np.argmax(torch.exp(preds).numpy(), axis=1)
preds[:20]
# np.argmax(torch.softmax(preds, dim=1).numpy(), axis=1)

# ### choose specific example where we want to evaluate sensitivity

test_index = 0 # a collapse case
# test_index = 3 # a noncollapse case

Xtrue = X_test.to_numpy()[test_index].copy()

# +
# Xtrue

# +
bounds = np.zeros((nFeatures,2))
bounds[:,0] = -0.2*np.abs(Xtrue)
bounds[:,1] = 0.2*np.abs(Xtrue)

bounds = bounds.tolist()
# -

# Define the model inputs
problem = {
    'num_vars': nFeatures,
    'names':  featureSets[feature_set],
    'bounds': bounds
}

problem

# Generate samples
Xsample = saltelli.sample(problem, 1024)

Xsample.shape

Xsample[0]

Xtrue

Xtrue + Xsample[0]


def evalSensitivity(Xsample, Xtrue, fixColumns=None, model=None, output=None):
    
    X0 = Xtrue.copy()
    X0 = torch.Tensor(X0)
    
    print (Xsample.shape, Xtrue.shape)
    
    model.eval()

    p = model.forward(X0.to(DEVICE))
    m = p.squeeze() # means
    print (p.shape, m.shape)
    
    if output == 'probability':
        p_mean0 = torch.exp(m.cpu().detach()).numpy() # probability
        # p_mean0 = torch.softmax(m.cpu().detach(), dim=0).numpy() # probability
    elif output == 'logprob':
        p_mean0 = m.cpu().detach().numpy() # logsoftmax
    
    print(p_mean0.shape)

    outputs0 = np.array([p_mean0[1]])
    
    outputs = []
    for i in trange(len(Xsample)):
    
        X = Xtrue + Xsample[i]
        if fixColumns:
            X[fixColumns] = Xtrue[fixColumns]

        X = torch.Tensor(X)

        p = model.forward(X.to(DEVICE))
        m = p.squeeze() # means
        
        if output == 'probability':
            p_mean = torch.exp(m.cpu().detach()).numpy() # probability
            # p_mean = torch.softmax(m.cpu().detach(), dim=0).numpy() # probability
        elif output == 'logprob':
            p_mean = m.cpu().detach().numpy()  # logsoftmax
        
        out = np.array([p_mean[1]])
        
        outputs.append(out)
        
    outputs = np.asarray(outputs)

    return outputs, outputs0


Y, Y0 = evalSensitivity(Xsample, Xtrue, fixColumns=[], model=net, output=useOutput)


print (p.shape, Y.shape, Y0.shape)

Y, Y0

# +
# save sensitivity analysis to pickle file

sensitivity_dict = {'feature_set': feature_set, 
                  'Y': Y, 'Y0': Y0, 
                  'problem': problem,
                  'Xsample': Xsample, 'Xtrue': Xtrue}

# write python dict to a file
# output = open('sensitivity.pkl', 'wb')
# pickle.dump(sensitivity_dict, output)
# output.close()

# # read python dict back from the file
# pkl_file = open('sensitivity.pkl', 'rb')
# sensitivity_dict2 = pickle.load(pkl_file)
# pkl_file.close()

# print (sensitivity_dict2)

# -

Y.shape

# Perform analysis
Si = sobol.analyze(problem, Y[:,0], print_to_console=False)

# Print the first-order sensitivity indices
print(Si['S1'])

Si_df = Si.to_df()

Si_df[0]

Si_df[0].sort_values(['ST'],ascending=False)

# +
plt.rc('figure', figsize=(12, 4))

barplot(Si_df[0])
plt.grid(True)

plt.tight_layout()
# -

Si_df[1]

# +
plt.rc('figure', figsize=(12, 4))

barplot(Si_df[1])
plt.grid(True)

plt.tight_layout()
# -

Si_df[2]

# ## Delta Moment-Independent Measure


from SALib.sample import latin
from SALib.analyze import delta

# +
bounds = np.zeros((nFeatures,2))
bounds[:,0] = -0.2*np.abs(Xtrue)
bounds[:,1] = 0.2*np.abs(Xtrue)

bounds = bounds.tolist()
# -

# Define the model inputs
problem = {
    'num_vars': nFeatures,
    'names':  featureSets[feature_set],
    'bounds': bounds
}

# Generate samples
# nsamples = 10000
nsamples = 1024
Xsample = latin.sample(problem, nsamples) # This is Set 1

Xsample.shape

Y, Y0 = evalSensitivity(Xsample, Xtrue, model=net, output=useOutput)

Y.shape

# Perform analysis
results0 = delta.analyze(problem, Xsample, Y.squeeze(), print_to_console=True)


results0_df = results0.to_df()

# +
# Si_df['delta_significant'] = Si_df['delta_conf'] / Si_df['delta']
# Si_df['Sobol_significant'] = Si_df['S1_conf'] / Si_df['S1']
# -

results0_df

# +
plt.rc('figure', figsize=(12, 4))

barplot(results0_df)
plt.grid(True)

plt.tight_layout()

plt.savefig('sensitivity_delta_analysis.png')
# -

# Sort factors by importance
factors_sorted = np.argsort(results0['delta'])[::-1].tolist()
# factors_sorted = np.argsort(results0['S1'])[::-1].tolist()

factors_sorted

results_sorted_df = results0_df.iloc[factors_sorted,:].copy()

# +
plt.rc('figure', figsize=(12, 4))

barplot(results_sorted_df)
plt.grid(True)

plt.tight_layout()

# -

# ### print names of top features

print(results_sorted_df.index[:12].tolist())

# ### fix first feature and look at sensitivity to rest

Y, Y0 = evalSensitivity(Xsample, Xtrue, fixColumns=factors_sorted[0], 
                        model=net, output=useOutput)

# Perform analysis
results1 = delta.analyze(problem, Xsample, Y.squeeze(), print_to_console=True)


results1_df = results1.to_df()

# +
# Si_df['delta_significant'] = Si_df['delta_conf'] / Si_df['delta']
# Si_df['Sobol_significant'] = Si_df['S1_conf'] / Si_df['S1']
# -

results1_df

# +
plt.rc('figure', figsize=(12,4))

barplot(results1_df)
plt.grid(True)

plt.tight_layout()

plt.savefig('sensitivity_delta_analysis_fix_most_important.png')


# -

# ## loop through, replacing each important feature with default

def evalSensitivity2(Xsample, model=None, output=None):
    
    model.eval()

    outputs = []
    for i in range(len(Xsample)):
    
        X = Xsample[i]

        X = torch.Tensor(X)

        p = model.forward(X.to(DEVICE))
        m = p.squeeze() # means

        if output == 'probability':
            p_mean = torch.exp(m.cpu().detach()).numpy() # probability
            # p_mean = torch.softmax(m.cpu().detach(), dim=0).numpy() # probability
        elif output == 'logprob':
            p_mean = m.cpu().detach().numpy()  # logsoftmax
        
        out = np.array([p_mean[1]])
        
        outputs.append(out)
        
    outputs = np.asarray(outputs)

    return outputs


# +
X_Set1 = Xsample+Xtrue

Y1 = evalSensitivity2(X_Set1, model=net, output=useOutput)

results1 = delta.analyze(problem, X_Set1, Y1.squeeze(), print_to_console=True)
results1 = results1.to_df().to_numpy()

# +
# Sort factors by importance
factors_sorted = np.argsort(results0['delta'])[::-1]
 
# Set up DataFrame of default values to use for experiment
X_defaults = np.tile(Xtrue,(nsamples, 1))
 
# Create initial Sets 2 and 3
X_Set2 = np.copy(X_defaults)
X_Set3 = np.copy(X_Set1)
 
for f in range(1, len(factors_sorted)+1):
    ntopfactors = f
     
    for i in range(ntopfactors): #Loop through all important factors
        X_Set2[:,factors_sorted[i]] = X_Set1[:,factors_sorted[i]] #Fix use samples for important
        X_Set3[:,factors_sorted[i]] = X_defaults[:,factors_sorted[i]] #Fix important to defaults
     
        # Run model for all samples    
        Y2 = evalSensitivity2(X_Set2, model=net, output=useOutput)
        Y3 = evalSensitivity2(X_Set3, model=net, output=useOutput)

        # Calculate coefficients of correlation
        coefficient_S1_S2 = np.corrcoef(Y1.squeeze(),Y2.squeeze())[0][1]
        coefficient_S1_S3 = np.corrcoef(Y1.squeeze(),Y3.squeeze())[0][1]
    
        print(f,i, coefficient_S1_S2, coefficient_S1_S3)

# -
# ## captum: integrated gradients

from captum.attr import IntegratedGradients


ig = IntegratedGradients(net)

X = Xsample + Xtrue
# X = np.tile(Xtrue,(100,1))

X.shape

test_input_tensor = Tensor(X).to(DEVICE)

test_input_tensor.requires_grad_()
attr, delta = ig.attribute(test_input_tensor, target=0, return_convergence_delta=True)
attr = attr.cpu().detach().numpy()


# +
plt.rc('figure', figsize=(12, 6))

def visualize_importances(feature_names, importances, title="Average Integrated Gradients Importances", 
                          plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
#         plt.figure(figsize=(16,6))
        plt.bar(x_pos, importances, width=0.5, align='center')
        # plt.bar(x_pos, np.abs(importances), width=0.5, align='center')
        plt.xticks(x_pos, feature_names, rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
feature_names = featureSets[feature_set]
visualize_importances(feature_names, np.mean(attr, axis=0))

plt.savefig('sensitivity_integrated_gradients.png')
# -


# Sort factors by importance
factors_sorted = np.argsort(-np.abs(np.mean(attr, axis=0)))

np.array(feature_names)[factors_sorted].tolist()

visualize_importances(np.array(feature_names)[factors_sorted].tolist(),
                      np.abs(np.mean(attr, axis=0)[factors_sorted]))





