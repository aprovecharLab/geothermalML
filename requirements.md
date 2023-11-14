# requirements

Recommendations for setting up machine learning tools on a personal computer using  python, pytorch, and scikits-learn all within the jupyter lab environment.
 
## Prerequisites

#### What things you need to install the software and how to install them.

-   A personal computer preferably with an integral or external graphic processing unit (GPU). I am using both a Dell XPS-13 9370 Developer Edition laptop \[This machine has a  Thunderbolt 3 USB C port and an external NVIDIA GTX 1070Ti  GPU in an Alito Node Pro enclosure\] and a System-76 AMD laptop with an integrated GPU. Both are running Ubuntu 20.04 LTS operating systems. You can do the same with Microsoft Windows 10  installed or with Apple Macintosh Computers.

- A python programming language distribution of which the Anadonda distributions are highly recommended. 

- You can download Ubuntu Linux from:
https://ubuntu.com/download/desktop
    
- You can download the latest Anaconda distribution for python 3.8+ from: https://www.anaconda.com/products/individual#Downloads

- Once you have Anaconda python installed then installing pytorch and all else is straight forward as explained below.     

## Install

*these instructions are for linux, but in windows you can open an Anaconda console or terminal or even do some of these things within a GUI*

### First install the latest anaconda python 3.8+, open a terminal, and update

- you will notice that once you install Anaconda,  "(base)" appears before your terminal prompt as:

```console
(base) $
```
- to update the "base" environment do: 

```console
(base) $ conda update conda
(base) $ conda update --all
```
### Create a specific environment for your pytorch machine learning functions

*I use python environments for each type of project like this, so that if something becomes corrupted you do not mangle your base python environment*

- create a "yml" file for your environment. this specifies exactly which libraries and perhaps specific versions your python knows about. Look in folder "./tools/anaconda_environments/" for example yml files. As an example a workable pytorch environment file is called "environment-pytorch.yml" and is shown at the bottom of this document (for up-to-date instructions for pytorch see:  https://pytorch.org/get-started/locally/). Two methods of installation are used here (1) conda and (2) pip. At the top of this file you will see the name of the environment we use, the "channels" or sources of the libraries, and lists of all of the packages we include.  Of course feel free to add or subtract packages from this file. 

- create the environment itself with the command:

```console
$ conda env create -f environment-pytorch.yml
```

- activate your new environment, then just for fun deactivate again:

```console
(base) $ conda activate pytorch
(pytorch) $ conda deactivate
(base) $
```
*I have found that updating the pytorch environment often breaks things ... you might need to delete the environment occasionally and reinstall rather than updating within it ... some quirk about pytorch dependencies. You should be able to update within environments, but I've had trouble with pytorch in this regard.*

 - you can list all of your environments

```console
(base) $ conda info -e
```

 - if you add other environment make sure you are at "base"

 - later once you are sick of this, you can remove the environment and all of its contents

```console
(base) $ conda env remove --name pytorch
```

## Ready to go

 - activate your environment

```console
(base) $ conda activate pytorch
```

 - run jupyter lab

```console
(pytorch) $ jupyter lab
```

 - a new web browser page will open, if not open a browser and point to the URL displayed in this terminal

 - when you are all done - close your browser and kill the jupyter lab server by typing:

```console
(pytorch) $ ^c^c
```

---

## Running some programs

 - By defining the pytorch environment using the "environment-pytorch.yml" file we have made available all of the tools you will likely need: python, numpy, matplotlib, scikits-xxx, pytorch (including CUDA libraries for the GPU if you have one), and jupyter lab plus more. 


## Example environment file: environment-pytorch.yml

```yml
name: pytorch
# see https://pytorch.org/get-started/locally/

channels: 
  - pytorch
  - nvidia
  - conda-forge

dependencies:
  - python
  - numpy
  - scipy
  - scikit-learn
  - scikit-image
  - h5py
  - ipywidgets
  - jupyterlab
  - jupytext
  - pandas
  - matplotlib
  - seaborn
  - pillow
  - pytorch
  - torchvision
  - cudatoolkit=11.1
#
  - umap-learn
  - susi
  - somoclu
#
  - bokeh
#
  - captum
  - tqdm
  - arrow
#
  - gdal
  - rasterio
#
  - opencv
  - pywavelets
#
  - pip
  - pip:
    - pyro-ppl
    - SALib    
    - gpytorch
    - tables
    - torch-summary
    - kornia
    - imgaug
    - deap
    - acoustics
    - brancher
    - arviz
    - pymc3
    - gpytorch
    - desolver
    - blitz-bayesian-pytorch
    - minisom
    - quicksom
    - simpsom
```

