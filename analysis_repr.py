import numpy as np
import torch
import pathlib
import math

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import argparse
import requests
import zipfile


#*******************************************************************************
def compute_accuracy_chunk(n_trials, chunk_size, features, weights, bias, targets, criterion):

    W = features.shape[1]
    acc_av = np.empty(n_trials)
    loss_av = np.empty(n_trials)

    for i in range(n_trials):
        ind = np.random.choice(W, size = chunk_size, replace = False)
        outputs = torch.matmul(features[:, ind], weights[:, ind].T)+bias
        #loss with random subset of activations
        loss_av[i] = criterion(outputs, targets).item()
        #accuracy with random subset of activations
        if args.dataset=='cifar10':
            _, max_logit = outputs.max(1)
            accuracy = max_logit.eq(targets).sum().item()/len(targets)
        acc_av[i] = accuracy

    return acc_av, loss_av



def compute_r2_chunk(n_trials, features, w, compute_sym=False, norm = 'mean'):
    features = features.numpy()
    print(f'compute r2 w = {w}')
    r2_global = np.empty(n_trials)
    mean_corr = np.empty(n_trials)

    nfeatures = features.shape[1]
    chunk_size = w

    for i in range(n_trials):

        #select random chunk
        ind_tot = np.arange(nfeatures)
        perm = np.random.permutation(np.arange(nfeatures))
        ind1 = perm[:chunk_size]
        ch1 = features[:, ind1]

        #ridge regression
        reg = Ridge(1e-8).fit(ch1, features)

        #R2 coeffient
        r2_global[i] = r2_score(features, np.matmul(ch1, reg.coef_.T)+reg.intercept_, multioutput = 'variance_weighted')

        #mean correlation
        res = features - reg.predict(ch1)               # N x n_feat
        cov = np.cov(res.T)                             # n_feat x n_feat
        #normalization (regularized)
        norm = np.diag(cov)                             # n_feat
        norm = (np.array([norm]).T*norm)**0.5 + 1e-8    # n_feat x n_feat + regularization
        #residual correlation matrix
        corr = np.divide(cov, norm)
        #average upper triugular matrix
        nondiag_entries = corr[np.tril_indices_from(corr, k = -1)]
        mean_corr[i] = np.mean(abs(nondiag_entries))

    return r2_global, mean_corr


#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='DIR', default = './data')
parser.add_argument('--dataset', metavar='DIR', default = 'cifar10')
parser.add_argument('--r2_rep', type = int, default = 40)
parser.add_argument('--acc_rep', type = int , default = 100)
args = parser.parse_args()

#*******************************************************************************
#try to import the represenatations weights and biases and labels
try:
    ensemble_av = np.load(f'{args.path}/download_repr/ensamble_averages_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_n20_20_nrep10.npy')[0]

    features_test = torch.load(f'{args.path}/download_repr/features_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_seed12_test.pt')
    features_train = torch.load(f'{args.path}/download_repr/features_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_seed12_train.pt')

    targets_test = torch.load(f'{args.path}/download_repr/cifar10_labels_test_set.pt')
    targets_train = torch.load(f'{args.path}/download_repr/cifar10_labels_training_set.pt')

    weights = torch.load(f'{args.path}/download_repr/weights_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_seed12.pt')
    bias = torch.load(f'{args.path}/download_repr/bias_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_seed12.pt')

except:
    raise FileNotFoundError("Files not found. Run the 'download.py' script or check the path")



width_tot = 512             #number of activations in the second to last representation of WideResNet28 on CIFAR10
chunk_size = np.sort(
        width_tot//np.array([2**i for i in range( int(math.log2(width_tot)) +1 )])
        )                                                                       #chunk sizes

pathlib.Path(f'{args.path}/results').mkdir(parents=True, exist_ok=True)         #path to save the results


"computation of chunck training and test accuracy"
criterion = torch.nn.CrossEntropyLoss()
acc_train = []
acc_test = []
print(f'evaluating accuracy chunks')
for c in chunk_size:
    acc_, loss_  = compute_accuracy_chunk(n_trials=args.acc_rep, chunk_size=c, features=features_test,
                                weights=weights, bias=bias, targets=targets_test, criterion=criterion)
    acc_test.append(np.mean(acc_))

    acc_, loss_  = compute_accuracy_chunk(n_trials=args.acc_rep, chunk_size=c, features=features_train,
                                weights=weights, bias=bias, targets=targets_train, criterion=criterion)
    acc_train.append(np.mean(acc_))

acc_train = np.array([chunk_size, np.array(acc_train)])
acc_test = np.array([chunk_size, np.array(acc_test)])


np.save(f'{args.path}/results/acc_chunk_c10_train_wr28_8.npy', acc_train)
np.save(f'{args.path}/results/acc_chunk_c10_test_wr28_8.npy', acc_test)



"computation of the R2 coefficiant and mean correlation"
r2 = []
mean_corr = []
print(f'evaluating r2 chunks')
for c in chunk_size:
    r2_, mean_corr_ = compute_r2_chunk(n_trials=args.r2_rep, features=features_test, w=c, compute_sym=False, norm = 'mean')
    r2.append(np.mean(r2_))
    mean_corr.append(np.mean(mean_corr_))
r2 = np.array([chunk_size, np.array(r2)])
mean_corr = np.array([chunk_size, np.array(mean_corr)])

np.save(f'{args.path}/results/r2_c10_test_wr28_8.npy', r2)
np.save(f'{args.path}/results/mean_corr_c10_test_wr28_8.npy', mean_corr)
