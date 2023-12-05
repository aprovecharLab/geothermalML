# Requirements

Here are some recommendations for setting up machine learning tools on a personal computer using  python, pytorch, and scikits-learn all within the jupyter lab environment.
 
## Prerequisites

#### What things you need to install the software and how to install them.

-   A personal computer preferably with an integral or external graphic processing unit (GPU). In December 2023 I am using both a Dell XPS-13 9370 Developer Edition laptop \[This machine has a  Thunderbolt 3 USB C port and an external NVIDIA GTX 1070Ti  GPU in an Alito Node Pro enclosure\] and a System-76 AMD laptop with an integrated GPU. Both are running Ubuntu 23.10 operating systems. You can do the same with Microsoft Windows 10 or 11 installed or with Apple Macintosh Computers.

- You can download Ubuntu Linux from:
https://ubuntu.com/download/desktop.
    
- A python programming language distribution of which the Mamba or the Anadonda distributions are highly recommended. 

- I am currently using the Mamba distribution of python. You can read about how to install the latest Mamba distribution for python here
https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html.

## Install

The file in this git archive "tools/install_mamba_notes.txt" describes the specific instructions for installation and set-up of the python+pytorch environment as used for this project. This creates a pytorch-specific python environment for geoscience machine learning called pytorchGeo. The "yml" file specifying this environment is appended below.

*I use python environments for each type of project like this, so that if something becomes corrupted you do not mangle your base python environment.*

*These instructions are for linux, but in windows you can open an python console or terminal or even do some of these things within a GUI.*

### Activate your new environment, then just for fun deactivate again:

```console
(base) $ mamba activate pytorchGeo
(pytorchGeo) $ mamba deactivate
(base) $
```
*I have found that updating the pytorch environment often breaks things ... you might need to delete the environment occasionally and reinstall rather than updating within it ... some quirk about pytorch dependencies. You should be able to update within environments, but I've had trouble with pytorch in this regard.*

 - You can list all of your environments:

```console
(base) $ mamba info -e
```

 - If you add other environment make sure you are at "base".

 - Later once you are sick of this, you can remove the environment and all of its contents:

```console
(base) $ mamba env remove --name pytorchGeo
```

## Ready to go

 - Activate your environment:

```console
(base) $ mamba activate pytorchGeo
```

 - Run jupyter lab:

```console
(pytorchGeo) $ jupyter lab
```
or use a different browser from the system default, e.g.,
``` console
(pytorchGeo) $ jupyter lab --browser=firefox
```

 - A new web browser page will open, if not open a browser and point to the URL displayed in this terminal.

 - When you are all done - close your browser and kill the jupyter lab server by typing:

```console
(pytorchGeo) $ ^c^c
```

---

## Running some programs

 - By defining the pytorch environment using the "environment-pytorchGeo.yml" file we have made available all of the tools you will likely need: python, numpy, matplotlib, scikits-xxx, pytorch (including CUDA libraries for the GPU if you have one), and jupyter lab plus more. 

 - You can run the jupyter notebooks (*.ipynb) in the folder "tests" to test your pytorch installation and whether or not the GPU on your computer functions properly.


## Example environment file as described in "tools" folder: environment-pytorchGeo.yml

```yml
name: pytorchGeo
# see https://pytorch.org/get-started/locally/

# to use: perform two steps ... make a jupyterlab barebones env, then activate and update with this yml file
# bash Mambaforge-$(uname)-$(uname -m).sh
# mamba create -n pytorchGeo jupyterlab -c conda-forge
### mamba activate pytorchGeo
# cd Desktop/AnacondaEnvironments/pyTorch/
# mamba env update -f environment-pytorchGeo.yml

channels: 
  - conda-forge
  - pytorch
  - nvidia
  - gpytorch
  # - fastchan

dependencies:
  - python
  - numpy
  - scipy
  - scikit-learn
  - scikit-image
  - h5py
  - pytables
  # - jupyter ### jupyter notebook only
  # - jupyterlab ### seems this might cause trouble when updating
  - jupytext
  - ipywidgets
  - ipympl
  - pandas
  - matplotlib
  - seaborn
  - pillow
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit=11.8
#
  - bokeh
#
  - umap-learn
#
  # - susi
  # - somoclu
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
  - pycwt
#
  - cython
  - pyproj
  - shapely
  - rtree
  - pyorbital
#
  - sympy
#
  - astropy
#
  - tabulate
#
  - gpytorch
  # - gpy
  # - botorch
  # - fastai
#
  - opentsne
#
  - tslearn
#
  - pip
  - pip:
    - pyro-ppl
    - SALib    
    - torch-summary
    - kornia
    - imgaug
    - pygeotools
    - pygeoprocessing
    - pyrtools
    - earthpy
    - filterpy
    - acoustics
    - deap
    - arviz
    - desolver
    # - tables ### this is same as pytables installed above
    # - emukit
    # - minisom
    # - quicksom
    # - simpsom
```
