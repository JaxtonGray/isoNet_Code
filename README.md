Deep Learning Thesis - IsoNet
==============================
This repository consists of the code I used to build the deep learning neural network for my thesis. Lovingly called 'IsoNet' here and within the code for simplification. The aim of this thesis is to examine whether or not it was capalble of building an deep learning model to estimate precipitable stable water isotopes here in Canada. 

The repository consists of the main code, the data, and the results. The main code is where the model was built, the data folder contains data and data cleaning, and the results folder was post-model analysis.
The following will be a guide to the files to check if you are interested in the code, data, or results.

## Code
There is one major file to examine here and that is the *isoNet_ModelCreation.ipynb* file. This is the file where the model was built and trained. The file is a Jupyter Notebook file and can be opened in Jupyter Notebook, or depending on the IDE you are using, it can be opened there as well. There is documentation within the file to help guide you through the process of building the model.

## Data
The major file here is *Data Combination and Cleaning.ipynb*. This file is where the data was combined and cleaned. It is worth noting that while some data is included in the repository, the HydroGFD data is not as the netcdf files are too large to be included. Within the *Data Combination and Cleaning.ipynb* file, there is a section that explains how to download the data from the HydroGFD website.

## Results
There are 2 files to examine here. The first is *Diagnostics.ipynb* and the second is *Statistical Analysis.ipynb*.

The *Diagnostics.ipynb* file is where attributes of the results where examinded to understand the model itself, with some comparisons to isoP. 

The *Statistical Analysis.ipynb* file is where the results were compared to the actual data to see how well the model was performing.
