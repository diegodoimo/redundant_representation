# Redundant representation help generalization

Source code of the paper [Redundant represenatation help generalization in wide neural networks](https://openreview.net/pdf?id=lC5-Ty_0FiN)
published at NeurIPS 35 


## Platforms:
- Ubuntu 22.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html. Then install the dependencies shown below manually.

```
conda create -n redundant_repr                                #create empy environment named "redundant_repr"
conda activate redundant_repr
conda install python numpy matplotlib seaborn scikit-learn    #install relevant python packages
conda install pytorch cpuonly -c pytorch          
```
Alternatively, you can create the environment with all the required dependencies through the .yml file by typing:
```
conda env create -f redundant_repr.yml
```

## Download data and reproduce plots of CIFAR10 on Wide-ResNet28_8

Download the representations of the second-to-last layer of Wide-ResNet28_8 __trained__ on CIFAR10. The data is saved in "./data/download_repr":

```
python download.py 
```

The _analysis_repr.py_ script allows to:
* compute the test error of the chuncks (Fig. 3.b); 
* compute the training error of the chunks (Fig. 4.b);
* compute the $R^2$ coefficient of fit of the chunks to the full-layer representation and their 'mean correlation' (Fig. 4.e). 

The results are saved in "./data/results". The '--r2_rep' and '--acc_rep' arguments allow to set the number of chunks on which the r2/mean_corr and accuracies are averaged. The profiles of the figures shown below can be made smoother by increasing the number of repetitions (with --r2_rep --acc_rep) on which the statistics are computed:

```
python analysis_repr.py --r2_rep 40 
```

Plot the results. The plots are saved in "./plots":
```
python plot_results.py
```

## Figures of  CIFAR10 on Wide-ResNet28-8

![Alt text](plots/cifar10_wr28_plots.jpg)

## Train the netowork from scratch and estract the representations

### _This part of repository is under development_
