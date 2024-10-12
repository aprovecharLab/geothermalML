# Brief Installation Notes
---

### This is a description for setting up a python environment for geoscience machine learning using pytorch and other useful libraries. 

### The environment is named "pytorchGeo".

### This makes use of "mamba," a Python-based CLI conceived as a drop-in replacement for conda as was originally used in the "anaconda" python distribution.

### For details see: `https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html`.

### For the latest information about installing the latest pytorch version see: `https://pytorch.org/get-started/locally/`.

---

## To install the environment perform two steps: first make a barebones jupyterlab environment, then update it with the yml file.

### From the command line in a terminal as an ordinary user with a prompt '$' do the following (choosing the default answers to questions):

`$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh`

`$ bash Miniforge3-$(uname)-$(uname -m).sh`

`$ mamba create -n pytorchGeo jupyterlab -c conda-forge`

### close the terminal and open a new one ... then go to folder with the yml file

`$ cd pythonEnvironments/Mamba/pyTorch/`

`$ mamba env update --name pytorchGeo --file environment-pytorchGeo.yml`
  
### Warning ... don't use the flag `--prune` above as it will remove jupyterlab, etc!
  
### You can now start the environment by typing:
  
`$ mamba activate pytorchGeo`

### Use the accompanying ymp file `./pytorch/environment-pytorchGeo.yml` to do the installation just described.

