# Purpose

### This git archive makes available a set of python-based machine learning modules for geothermal energy resource evaluation. 

# Origin

### The software available in this archive was developed by Dr. Stephen Brown (of the Massachusetts Institute of Technology and Aprovechar Lab L3C) while participating in two research projects hosted by the Great Basin Center for Geothermal Energy (GBCGE) of the University of Nevada, Reno.

## (1) The Nevada Geothermal Machine Learning Project 
(https://gbcge.org/current-projects/machine-learning/)

### GBCGE project personnel: 
Dr. James Faulds (PI); Dr. Bridget Ayling; Elijah Mlawsky; Dr. Cary Lindsey; Connor Smith; Dr. Mark Coolbaugh

### Project collaborators: 
United States Geological Survey; Hi-Q Geophysical; Massachusetts Institute of Technology

### Project duration: 
24 months: 1 August 2019 – 28 February 2022.

### Total project funding: 
$500,000

### Funding agency: 
U.S. Department of Energy Geothermal Technologies Office (award number DE-EE0008762).

### Project goal:
 Apply machine learning (ML) techniques to develop an algorithmic approach to identify new geothermal systems in the Great Basin region and build on the successes of the Nevada geothermal play fairway project. The reason for this is that an algorithmic approach that empirically learns to estimate weights of influence for diverse parameters may scale and perform better than the original workflow developed for play fairway analysis. Project activities include augmenting the number of training sites (positive and negative) that are needed to train the ML algorithms, transforming the data into formats suitable for ML, and development and testing of the ML techniques and outputs.


 # and 
 
## (2) INnovative Geothermal Exploration through Novel Investigations Of Undiscovered Systems (INGENIOUS)
(https://gbcge.org/current-projects/ingenious/)

### GBCGE project personnel:
- Dr. Bridget Ayling (PI); Dr. James Faulds (Co-PI);  Dr. Anieri Morales Rivera; Dr. Richard Koehler; Dr. Corné Kreemer; Elijah Mlawsky; Dr. Mark Coolbaugh; Rachel Micander; Craig dePolo; Kurt Kraal; Nicole Wagoner; Noah Williams; Quentin Burgess; Mary-Hannah Giddens; Chris Kratt; Chris Sladek

### Project collaborators: 
 -United States Geological Survey; Utah Geological Survey; Idaho Geological Survey; National Renewable Energy Laboratory; Lawrence Berkeley National Laboratory; Raser Power Systems; Geothermal Resources Group; Hi-Q Geophysical; Aprovechar Lab L3C; Petrolern Ltd.; Innovate Geothermal Ltd.; W Team Geosolutions.

### Project duration: 
4.5 years: 1 February 2021 – 30 June 2025.

### Total project funding: 
$10,000,000

### Funding agency: 
U.S. Department of Energy Geothermal Technologies Office (award number DE-EE0009254 ).

### Project goal: 
The primary goal of this project is to accelerate discoveries of new, commercially viable hidden geothermal systems while reducing the exploration and development risks for all geothermal resources. We stand at a major crossroads in geothermal research and development, whereby major achievements have been made in PFA, 3D and conceptual modeling, resource capacity estimation, ML, the application of advanced geostatistics, and value of information (VOI) analysis, but these techniques have yet to be combined into one over-arching best-practices workflow for a broad region. Our ambitious project proposes to fully integrate these techniques to develop a comprehensive exploration workflow toolkit that includes predictive geothermal PF maps at both the regional- and prospect-scale, detailed 3D maps and conceptual models, and a developers’ playbook. Building on geothermal play fairway (PF) efforts in central Nevada, NE California/NW Nevada, and western Utah, we will expand these study areas to the broader Great Basin region for early stage prospect identification. Concurrently, we will move several blind prospects forward with detailed geological and geophysical analyses followed by drilling thermal-gradient holes (TGH) and possibly slimholes.



# Description



## Archive Contents (directory tree)
```
Geothermal_ML_git_archive
├── datasets
│   └── files_location.txt
├── docs
│   ├── Brown_et_al-Bayesian_Neural_Networks_for_Geothermal_Resource_Assessment-arxiv.org:2209.15543.pdf
│   └── Brown_et_al-Machine_Learning_for_Natural_Resource_Assessment-GRC:1034262.pdf
├── LICENSE.md
├── modules
│   ├── ANN
│   │   ├── ANN_skeleton_inference_runSavedModel.ipynb
│   │   ├── ANN_skeleton_inference_runSavedModel.py
│   │   ├── ANN_skeleton_readPFAdataframe.ipynb
│   │   └── ANN_skeleton_readPFAdataframe.py
│   ├── BNN
│   │   ├── BNN_skeleton-alpha-loop-plots-VCdim.ipynb
│   │   ├── BNN_skeleton-alpha-loop-plots-VCdim.py
│   │   ├── BNN_skeleton_alpha_loop_readPFAdataframe.ipynb
│   │   ├── BNN_skeleton_alpha_loop_readPFAdataframe.py
│   │   ├── BNN_skeleton_inference_runSavedModel-stats.ipynb
│   │   ├── BNN_skeleton_inference_runSavedModel-stats.py
│   │   ├── BNN_skeleton-readPFAdataframe.ipynb
│   │   └── BNN_skeleton-readPFAdataframe.py
│   ├── data_preprocessing
│   │   ├── PFA_preprocessing.py
│   │   ├── PFA-preprocessing_test.ipynb
│   │   ├── PFA-preprocessing_test.py
│   │   ├── PFA_save_benchmarks.ipynb
│   │   └── PFA_save_benchmarks.py
│   ├── Gaussian_Processes_Regression
│   │   ├── GeochemT_Gaussian_Process_Regression.ipynb
│   │   ├── GeochemT_Gaussian_Process_Regression.py
│   │   ├── HF-residual_Gaussian_Process_Regression.ipynb
│   │   ├── HF-residual_Gaussian_Process_Regression.py
│   │   └── results
│   │       └── about.txt
│   └── SiameseNN
│       ├── BNN_model_trial_2.35.torch
│       ├── BNN_pretrained_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb
│       ├── BNN_pretrained_siamese_ContrastiveLoss_EuclidianDist-skeleton.py
│       ├── BNN_trainable_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb
│       └── BNN_trainable_siamese_ContrastiveLoss_EuclidianDist-skeleton.py
├── README.md
├── requirements.md
├── tests
│   ├── pytorch_test1.ipynb
│   ├── pytorch_test2.ipynb
│   └── pytorch_test3.ipynb
└── tools
    └── 000_pythonEnvironments
        └── Mamba
            ├── install_mamba_notes.txt
            └── pyTorch
                └── environment-pytorchGeo.yml

15 directories, 38 files
```



## Details 
The codes presented here are not turn-key applications, but rather a complete set of examples combined with an example (test) dataset. Users are encouraged to read the accompanying preprints and examine the codes themselves to gain familiarity with the approach and implementation.

### Below we describe the contents of this archive more explicitly.

---
#### Datasets
```
├── datasets
│   └── files_location.txt
```

This folder contains the example (test) dataset used for testing the codes. Since github has file size restrictions the text file here contains a hyperlink to the dataset.


---
#### Documents
```
├── docs
│   ├── Brown_et_al-Bayesian_Neural_Networks_for_Geothermal_Resource_Assessment-arxiv.org:2209.15543.pdf
│   └── Brown_et_al-Machine_Learning_for_Natural_Resource_Assessment-GRC:1034262.pdf
```
This folder contains the research papers written during this project and represent a textbook of the methods employed and the theory behind them for the initial Artificial Neural Network and Bayesian Neural Network implemented here. These models form the basis and provide context for the Siamese Neural Network and the Gaussian Process Regression Feature Engineering modules also provided in this archive. However, these last two modules are not described in these papers.

---
#### License
```
├── LICENSE.md
```

This file describes the license for use and distribution of this software.

---
#### Artificial neural networks
```
├── modules
│   ├── ANN
│   │   ├── ANN_skeleton_inference_runSavedModel.ipynb
│   │   ├── ANN_skeleton_inference_runSavedModel.py
│   │   ├── ANN_skeleton_readPFAdataframe.ipynb
│   │   └── ANN_skeleton_readPFAdataframe.py
```

These are the Artificial Neural Network modules. Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.

Here there are two modules:

-  The first module is used for training and saving a network is called "ANN_skeleton_readPFAdataframe." 

- The second used for reading the trained network, evaluating its properties, and making predictions and maps. It is called "ANN_skeleton_inference_runSavedModel."

The training module reads data prepared and preprocessed by the preprocessing module described later. The preprocessing module supplied here here makes use of the example dataset from Nevada USA. Following the example, modifying the preprocessing module allows a generic dataset to be created and custom preprocessing to be done for use of these modules in new geographic areas with possibly a different set of geological and geophysical features and a different set of training data and labels.

---
#### Bayesian neural networks
```
├── modules
│   ├── BNN
│   │   ├── BNN_skeleton-alpha-loop-plots-VCdim.ipynb
│   │   ├── BNN_skeleton-alpha-loop-plots-VCdim.py
│   │   ├── BNN_skeleton_alpha_loop_readPFAdataframe.ipynb
│   │   ├── BNN_skeleton_alpha_loop_readPFAdataframe.py
│   │   ├── BNN_skeleton_inference_runSavedModel-stats.ipynb
│   │   ├── BNN_skeleton_inference_runSavedModel-stats.py
│   │   ├── BNN_skeleton-readPFAdataframe.ipynb
│   │   └── BNN_skeleton-readPFAdataframe.py
```

These are the Bayesian Neural Network modules. Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.

There are four modules:

- The first module is used for training and saving a network. It is called "BNN_skeleton-readPFAdataframe.ipynb." 

- The second module is used for reading the trained network, evaluating its properties, and making predictions and maps. It is called "BNN_skeleton_inference_runSavedModel-stats.ipynb." 

- The third module is for looping on the key hyperparameter which balances data fit and model complexity and saving some key metrics from these trials. It is called "BNN_skeleton_alpha_loop_readPFAdataframe.ipynb." 

- The fourth module is for reading the hyperparamter metrics, creating some diagnostic plots and implements methods help choose an optimal hyperparameter value. This module is called "BNN_skeleton-alpha-loop-plots-VCdim.ipynb." Once the optimal hyperparameter is chosen, the training module can be run to create and save a "best" model for predictions.

The training module reads data prepared and preprocessed by the preprocessing module described later. The preprocessing module supplied here here makes use of the example dataset from Nevada USA. Following the example, modifying the preprocessing module allows a generic dataset to be created and custom preprocessing to be done for use of these modules in new geographic areas with possibly a different set of geological and geophysical features and a different set of training data and labels.

Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.



---
#### data preprocessing
```
├── modules
│   ├── data_preprocessing
│   │   ├── PFA_preprocessing.py
│   │   ├── PFA-preprocessing_test.ipynb
│   │   ├── PFA-preprocessing_test.py
│   │   ├── PFA_save_benchmarks.ipynb
│   │   └── PFA_save_benchmarks.py
```

These are the data preprocessing modules. Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.

The purpose of these modules is to take a raw tabular dataset and extract the relevant information from that file (features, labels, and essential auxiliary information such as geographic coordinates) for training and predictions, prescale and transform these as needed, separate the training examples from the rest and then create a python dictionary of the results. This dictionary is then saved to a python pickle archive. This preprocessed dataset is in the form needed for the neural network modules described above. As such the dataset produced is "generic" and the preprocessing codes can be modified to taste.

An essential preprocessing step is that of standard scaling, where the mean of each feature is removed and the values are normalized by the their standard deviation. Other things that can be don are to transform the data probability distribution to mitigate skewness and to transform categorical data to a numerical form. These operations are application specific so the user needs to evaluate the bets approach and modify the preprocessor accordingly.

There are three modules:

- The preprocessing functions themselves are implemented in the file "PFA_preprocessing.py." This set of functions is used by the other two modules.

- The working module which reads the raw tabular dataset, performs the preprocessing itself, and creates the output pickle file is called "PFA_save_benchmarks.ipynb." This module creates the input data files for training and prediction by the neural networks described earlier.

- The third module allows the user to see the consequences of the preprocessing through graphical means. This module is called "PFA-preprocessing_test.ipynb." The module reads the raw data file, performs preprocessing, and implements various displays of the results. This helps the user to adjust or to develop and test new candidate preprocessing algorithms.


---
#### Gaussian processes regression
```
├── modules
│   ├── Gaussian_Processes_Regression
│   │   ├── GeochemT_Gaussian_Process_Regression.ipynb
│   │   ├── GeochemT_Gaussian_Process_Regression.py
│   │   ├── HF-residual_Gaussian_Process_Regression.ipynb
│   │   ├── HF-residual_Gaussian_Process_Regression.py
│   │   └── results
│   │       └── about.txt
```

These are the modules for Gaussian Processes Regression. Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.

These modules are designed for the engineering of new physics-rich feature layers for training and prediction. 
The new layers are derived from new data types or measurements that may not  align in space with the main positive and negative training examples. The algorithm used for this feature engineering is Gaussian processes regression which may also be known as or at least has similarities to "Kriging" as used in geostatistics. Useful references for the method are available here: 

- http://gaussianprocess.org/gpml/chapters/RW.pdf

- https://scikit-learn.org/stable/modules/gaussian_process.html

The philosophy behind this is as follows. In the use of a fully-connected artificial neural network (either an ANN or a BNN) a set of input features are combined through nonlinear functions in each layer of the network. The input of each neuron in each layer, specifically the first layer, can be thought of a "engineered features since the components are combined in such a way with pre-multiplying weights (determined through optimization) as to better represent the physics and allow the network to better separate the examples into the target classes. 

In the past, engineering of the features as input to a neural network was often done by hand so as to introduce expert knowledge and to make the features better constrain the physics of the problem. We borrow from this idea and automate the process through Gaussian processes regression. Through this approach we take, say 10 features, combine them in a constrained manner using some new training information to create an 11th feature which is representative of the physics and the neural network is trained on the this enhanced feature set.

Here we provide two examples of this process: one by engineering a new feature layer by considering heatflow residual data, the second by creating a new feature layer by considering maximum subsurface water temperatures indicated by geothermometry. Both of these datasets are known as scalar values (real numbers) and locations within the study area which do not coincide with the locations of the known positive and negative training labels, nor are they known at any arbitrary pixel in the study area.

It would be possible to interpolate these values (heatflow residual and geochemistry temperature) using spatial Kriging by a regression on the geographic coordinates. We do not take that approach here. Instead we do a nonlinear regression in the high-dimensional feature space that we have available for training the artificial neural networks (in the example here 10 features). These original features are known at every pixel, so spatial information is implicit in the regression. 

It is assumed that first the regression target (e.g. heat flow residual) is a key indicator of our physical system which will allow us to predict the viability of a positive geothermal prospect. Second it is assumed that the base features have some intrinsic relation or capacity to predict the regression target itself (e.g. heat flow residual). So we combine the base features to interpolate the regression target as a scalar value, predict it everywhere, then use it as a newly engineered feature for classification with the artificial neural networks.

The codes for performing this process are:

- "HF-residual_Gaussian_Process_Regression.ipynb" for heat flow residuals.

-  "GeochemT_Gaussian_Process_Regression.ipynb" for geochemistry temperatures. 

- Auxiliary datasets for these examples are provided in the data archive described earlier. 

- An empty directory, "results," is provided as a place for the working plots.

The codes first read both the original and new datasets, extract the usable information from each as needed for the regressions, perform the regressions, and finally the results are plotted and saved. 

There are a few things to note. First there may be a need for custom preprocessing of the new regression target dataset, such as standard scaling and probability transformations. Examples of these are shown. Second there are hyperparameters that need to be carefully chosen for the regression algorithm. The consequences of these choices may be difficult to discern, so we provide some powerful dimensionality reduction algorithms and associated plots to allow the user to visualize the results and make the appropriate decisions. Useful references for the UMAP dimensionality reduction methods are available here:

- https://pypi.org/project/umap-learn/ 

- https://arxiv.org/abs/1802.03426


---
#### Siamese neural networks for site-by-site comparison
```
├── modules
│   └── SiameseNN
│       ├── BNN_model_trial_2.35.torch
│       ├── BNN_pretrained_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb
│       ├── BNN_pretrained_siamese_ContrastiveLoss_EuclidianDist-skeleton.py
│       ├── BNN_trainable_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb
│       └── BNN_trainable_siamese_ContrastiveLoss_EuclidianDist-skeleton.py
```

These are the modules for Siamese Neural Networks for site-by-site comparison. Generally, two separate files are provided for each purpose: (1) a python-based jupyter notebook *.ipynb and (2) a plain python version with markdown *.py . Both are functionally equivalent.

We borrow from image recognition and facial recognition applications to develop a geothermal energy prospect geographic site-by-site comparison using Siamese neural networks. The basic idea is that a neural network is run twice for each case (or in a sense making an interconnected twin, thus the reference to Siamese twins). One of the two sites is considered the reference, the other site(s) are tests. 

For a fully-connected neural network classifier each layer represents, in effect, an "engineered" feature set that has been created through non-linear combination of the original features in such a way as to better separate the training classes.

So, more explicitly, the set of geological and geophysical features for two separate geographic sites are fed part-way through the network to generate a set of salient "engineered" features that are then compared one set to another. These two engineered feature sets define two vectors in a high dimensional space. A metric describing the distance between these two vectors is calculated and represented as a "similarity" between the two sites. 

Given a reference site, then, the remaining pixels in the study area map can be compared and a new map of similarity to the reference can be produced. The intersection of this map with, say, the map of the regional resource potential probability generated by a Bayesian neural network, for example, may provide additional evidence for site selection.

Here two modules are provided: 

- The first module uses a pretrained Bayesian neural network optimized previously for defining resource probability through use of a training set. This module is called "BNN_pretrained_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb." Once trained, internal layers of this network can be utilized in a Siamese neural network forward calculation. Note that there are several options for defining the distance or similarity of the two sites. Here, the Euclidian distance between the vectors is used as the basis for this calculation. The user should consider these carefully for best use in their specific application.

- The second module attempts to train a Siamese Bayesian neural network from scratch, specifically optimized for the purpose of Siamese neural network similarity calculations. This module is called "BNN_trainable_siamese_ContrastiveLoss_EuclidianDist-skeleton.ipynb." This version uses the same training set as the BNN models, where now all of the geothermal benchmark training examples are fed to the twins in pairs (++, +-, -+, and --) for comparison. This module requires use of a different type of loss function of optimization. Contrastive loss is used here 
which requires some important hyperparameters to be chosen. Other options are available in the literature.

Some detailed discussions of Siamese neural networks and options for the loss function used during training are available here:

- https://dl.acm.org/doi/10.5555/2987189.2987282

- https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18

- https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec

- https://neptune.ai/blog/content-based-image-retrieval-with-siamese-networks

