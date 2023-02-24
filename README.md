# Redundant representation help generalization source code

Code of the paper [Redundant represenatation help generalization in wide neural networks](https://arxiv.org/abs/2106.03485)
published at NeurIPS 35 


Platforms:
- Ubuntu 22.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html, or install the dependencies shown below manually.

```
conda create -n redundant_repr                                #create empy environment named "redundant_repr"
conda activate redundant_repr
conda install python numpy matplotlib seaborn scikit-learn    #install relevant python packages
conda install pytorch cpuonly -c pytorch          
```

## Download data and reproduce plots of CIFAR10 on Wide-ResNet28_8.

Download the CIFAR10 representations of the second-to-last layer of WIde-ResNet28_8:

```
python download.py
```

Compute the test/training error of the chuncks, the R^2 coefficient of fit of the chunks to the full-layer representation, and the mean correlation of the chunks:

```
python analysis_repr.py
```

Plot the results:

```
python plot_results.py
```

![alt text](https://github.com/diegodoimo/redundant_representation/tree/master/plots/cifar10_wr28_plots.jpg?raw=true)

### _The repository is under development_
