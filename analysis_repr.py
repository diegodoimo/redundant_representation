import numpy as np
import torch
import sys
from dadapy.data import Data

from torchvision.datasets.vision import VisionDataset
from torch import nn
import torchvision.datasets as datasets
import os
import pathlib
import torchvision.transforms as transforms
import math

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
#import torch.nn.functional as F

#my_modules
sys.path.insert(0, '/home/diego/ricerca/packages/net_packages')
from my_nets.biroli_scaling import fully_connected, fully_connected_old, fully_connected_new
from NNmodules import layer_loader, activations_utils, my_dataset



#pre nips22_rev
#sys.path.insert(0, '/home/diego/ricerca/trained_models/training_scripts/icml2022_reviews/icml_rev_utils')
#sys.path.insert(0, "/home/diego/Documents/dottorato/ricerca/trained_models/training_scripts/cifar/utils")
#sys.path.insert(0, '/home/diego/ricerca/trained_models/training_scripts/icml2022_reviews/icml_rev_utils')
#from wide_resnet import wide_resnet28
#from densenet import densenet28, densenet40


#nips22 rev
sys.path.insert(0, "/home/diego/Documents/dottorato/ricerca/trained_models/training_scripts/cifar/utils")
from densenet import densenet40
from wide_resnet import wide_resnet28


#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'cifar10', metavar='DIR')
parser.add_argument('--dataset_path', metavar='DIR', default = '/home/diego/ricerca/datasets')
parser.add_argument('--training_set', action = 'store_true')


parser.add_argument('--model_path', metavar='DIR', default = './models')
parser.add_argument('--model_name', metavar='DIR', default = 'densenet40')

parser.add_argument('--feaure_path', default = './features_weights/cifar10')
parser.add_argument('--feature_name', default = 'features_densnet40_w....')

parser.add_argument('--param_path', metavar='DIR', default = './features_weights/cifar10')
parser.add_argument('--weight_name', metavar='DIR', default = '.........')
parser.add_argument('--bias_name', metavar='DIR', default = '......')

parser.add_argument('--batch_size', default = 4, type = 'int')
parser.add_argument('--device', default = 'cuda')

parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--results_path', metavar = 'DIR', default = './models/trials')
parser.add_argument('--resume_folder', metavar = 'DIR', default = None)
parser.add_argument('--resume_filename', metavar = 'DIR', default = None)
parser.add_argument('--filename')

args = parser.parse_args([])

#*******************************************************************************

def get_model(self, w, seed, nlay, sched, cpt = None):

    if self.data_name=='cifar10' or self.data_name=='cifar100' or self.data_name=='normal':

        if w == 16 and self.net_name=='wide_resnet28':
            net_name_tmp = self.net_name+'_1_inpl4'
            inplanes = 4
            w = 1

        elif w == 32 and self.net_name=='wide_resnet28':
            net_name_tmp = self.net_name+'_1_inpl8'
            inplanes = 8
            w = 1

        elif self.net_name == 'wide_resnet28':
            net_name_tmp = self.net_name+f'_{w//64}'
            inplanes = 16
            w = w//64
        elif self.net_name.startswith('densenet28'):
            net_name_tmp = self.net_name+f'_{int(w/3.75)}'
            inplanes = int(w/3.75)
        elif self.net_name.startswith('densenet40'):
            net_name_tmp = self.net_name+f'_{int(w/5.5)}'
            inplanes = int(w/5.5)

        if self.data_name == 'cifar10' or self.data_name =='normal': nc=10
        elif self.data_name == 'cifar100': nc=100

        if self.net_name =='wide_resnet28':
            model = wide_resnet28(num_classes=nc,
                                widening_factor = w,
                                inplanes = inplanes,
                                #last_layer_width = self.llw, #only used is a review not implemented for nips22
                                kernel_size = self.kernel_size)
        elif self.net_name=='densenet28':
            model = densenet28(num_classes=nc,num_init_features=inplanes, growth_rate = inplanes//2)
        elif self.net_name=='densenet40':
            model = densenet40(num_classes=nc,
                            num_init_features=inplanes,
                            growth_rate = inplanes//2,
                            #out_features = self.llw,       #only used is a review not implemented for nips22
                            kernel_size = self.kernel_size)

            if self.filename=='stem_type_imagenet':
                model = densenet40(num_classes=nc,num_init_features=inplanes, growth_rate = inplanes//2, stem_type='imagenet')


        if self.amp:
            filename = f'{self.data_name}_{net_name_tmp}_ep200_seed{seed}_wd{self.wd}_ls{self.ls_magnitude}_amp'
        else:
            #filename = f'{self.data_name}_{net_name_tmp}_ep200_optSGD_sched{sched}_seed{seed}'
            filename = f'{self.data_name}_{net_name_tmp}_ep200_seed{seed}'

        if not self.amp:
            #if self.label_smoothing:
            #    filename += f'_label_smoothing'
            #    print('label_smoothing')
            if self.wd is not None:
                filename+=f'_wd{self.wd}'
            if self.ls_magnitude is not None:
                filename+=f'_ls{self.ls_magnitude}'
            if self.llw:
               filename = filename+f'_llw{self.llw}'

        if self.filename is not None:
            filename = filename+f'_{self.filename}'

        if cpt is not None:
            path = f'{self.folder}/{filename}_model_{cpt}_cpt.pth.tar'
        else:
            path = f'{self.folder}/{filename}_model_best.pth.tar'

        if self.model_name is not None:
            path = f'{self.folder}/{self.model_name}'

        if self.state !='random':
            dict = torch.load(path, map_location = 'cpu')
            model.load_state_dict(dict['state_dict'])
        else:
            print('loading untrained model')

        if self.net_name=='wide_resnet28':
            weights = model.state_dict()['fc.weight'].to('cpu')
            bias = model.state_dict()['fc.bias'].to('cpu')
        elif self.net_name.startswith('densenet'):
            weights = model.state_dict()['classifier.weight'].to('cpu')
            bias = model.state_dict()['classifier.bias'].to('cpu')
        if self.net_name =='wide_resnet28':
            modules, names, depths = layer_loader.getWideResNetDepths(model, red = 10)
        elif self.net_name.startswith('densenet'):
            modules, names, depths = layer_loader.getdensenet(model, red = 10)


    elif self.data_name=='pmnist':
        model = fully_connected_old(width = w, depth = nlay)
        path =  f'{self.folder}/W{w}_seed{seed}_nlay{nlay}_best.pth.tar'

        dict = torch.load(path)
        model.load_state_dict(dict['state_dict'])
        weights = model.state_dict()['linear_out.weight'].to('cpu')
        bias = model.state_dict()['linear_out.bias'].to('cpu')

        modules, names, depths = layer_loader.get_fully_connected_biroli_depths(model)

    elif self.data_name=='pmnist_wd':
        model = fully_connected_new(w, depth = nlay, batch_norm = True, dropout = False)
        if cpt is not None:
            path =  f'{self.folder}/W{w}_seed{seed}_nlay6_wd{self.wd}_lr0.001_batch_norm_bs{self.bs}_sgd_{cpt}_cpt.pth.tar'
        else:
            path =  f'{self.folder}/W{w}_seed{seed}_nlay6_wd{self.wd}_lr0.001_batch_norm_bs{self.bs}_sgd_best.pth.tar'
        dict = torch.load(path)
        model.load_state_dict(dict['state_dict'])
        weights = model.state_dict()['linear_out.weight'].to('cpu')
        bias = model.state_dict()['linear_out.bias'].to('cpu')

        modules, names, depths = layer_loader.get_fully_connected_biroli_depths(model)

    return model, modules, weights, bias, names



def get_dataset(args):
    if args.dataset == 'cifar10':
        transform_val = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))]
                        )
        dataset = datasets.CIFAR10(f'{args.data_path}', train = args.training_set, transform = transform_val, download = True)

    elif args.dataset == 'cifar100':
        transform_val = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        dataset = datasets.CIFAR100(f'{args.data_path}', train = args.training_set, transform = transform_val, download = True)

    targets = torch.tensor(dataset.targets)
    targets = targets.type(torch.long)
    return dataset

def get_features(w, seed, nlay = 6, sched ='CosineAnnealingLR', cpt = None):

    with torch.no_grad():
        print(f'estracting features {self.net_name}_{w},  seed = {seed}' )

        dataset = get_dataset(args)
        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=10,
                                            pin_memory=True)

        model, modules, self.weights, self.bias, names = get_model(w, seed, nlay = nlay, sched = sched, cpt=cpt)

        model.to(args.device)
        model.eval()

        label = self._get_label(cpt = cpt, seed = seed)
        embdims = activations_utils.get_layer_depths(phantom, model, modules, self.device)
        nsamples = len(dataset)

        for l, module in enumerate(modules):
            if l==len(modules)-2:
                out = torch.empty(nsamples, embdims[l])
                if self.net_name.startswith('densenet'):
                    out = torch.empty(nsamples, embdims[l]//64)
                    out = activations_utils.get_activations(loader, model, self.device, out, batch_size, nsamples, module, postproc=True)
                else:
                    out = activations_utils.get_activations(loader, model, self.device, out, batch_size, nsamples, module)
                self.features = out.to('cpu')
                if self.trainset:
                    torch.save(self.features, f'{self.results_features}/features_{self.net_name}_{w}_{label}_train.pt')
                else:
                    torch.save(self.features, f'{self.results_features}/features_{self.net_name}_{w}_{label}_test.pt')
        return out




def compute_accuracy(features, weights, bias, n_trials, seed=0, cpt = None):

    chunk_size = np.sort(width_tot//np.array([2**i for i in range( int(math.log2(width_tot)) +1 )]))
    criterion, targets = self._get_loss_targets(width_tot=width_tot, seed=seed)
    label = self._get_label(cpt = cpt, seed = seed)

    for w in chunk_size:
        print(f'chunk_size = {w}')
        acc_av, loss_av = compute_accuracy_chunk(n_trials, w, features, weights, bias, targets, criterion)
        print(f'acc_av = {np.mean(acc_av): .4f}, loss_av = {np.mean(loss_av): .4f}')

        if self.trainset:
            np.save(f'{self.results_acc}/acc_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_train', acc_av)
            np.save(f'{self.results_loss}/loss_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_train', loss_av)
        else:
            print('saving...')
            np.save(f'{self.results_acc}/acc_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_test', acc_av)
            np.save(f'{self.results_loss}/loss_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_test', loss_av)


def compute_accuracy_chunk(self, n_trials, chunk_size, features, weights, bias, targets, criterion):
    print(f'w = {w}')
    W = featurs.shape[1]
    acc_av = np.empty(n_trials)
    loss_av = np.empty(n_trials)

    for i in range(n_trials):
        ind = np.random.choice(W, size = w, replace = False)
        outputs = torch.matmul(features[:, ind], weights[:, ind].T)+bias
        #loss with random subset of activations
        loss_av[i] = criterion(outputs, targets).item()
        #accuracy with random subset of activations
        accuracy = self._compute_accuracy(outputs, targets)
        acc_av[i] = accuracy

    return acc_av, loss_av

def compute_r2_corr(self, features, weights, bias, n_trials, seed = 0, cpt = None, nimg_class = None, save = True, filename = ''):

    chunk_size = np.sort(int(width_tot)//np.array([2**i for i in range( int(math.log2(width_tot)) +1 )]))

    print(f'chunk_sizes = {chunk_size}')
    if self.data_name != 'normal':
        subset = self._reduce_indices(nimg_class)
        features = self.features[np.array(subset)]

    ndata = len(features)

    for w in chunk_size:
        print(f'chunk_size = {w}' )
        r2, mean_corr = self._compute_r2_chunk(n_trials, features, w, compute_sym= compute_sym, norm = 'mean')
        print(f'r2_global = {np.mean(r2): .4f}, mean_corr = {np.mean(mean_corr): .4f}')

        np.save(f'{self.results_r2}/r2_global_{self.net_name}_{width_tot}{filename}_w{w}_nrep{n_trials}_ndata{ndata//1000}k_test', r2)
        np.save(f'{self.results_r2}/mean_corr_{self.net_name}_{width_tot}{filename}_w{w}_nrep{n_trials}_ndata{ndata//1000}k_test', mean_corr)



def _compute_r2_chunk(self, n_trials, features, w, compute_sym=False, norm = 'mean'):
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

















class stats_chunked_models:
    def __init__(self, dataset, model=None, trainset=False, device = 'cuda', ep=None, recompute_features = False, model_path = None,
                wd=None, bs = None, batch_norm = True, label_smoothing=False, ls_magnitude =None, llw=None, sgd= None, data_path = None,
                amp = False, filename=None, ablation= False, state = None, model_name= None, root_results=None, kernel_size = 3):

        self.data_name = dataset
        self.trainset = trainset
        self.device = device
        self.net_name = model #'fc_nlay6', 'wide_resnet28'
        self.batch_norm = batch_norm
        self.recompute_features = recompute_features
        self.label_smoothing = label_smoothing

        assert state is None or state=='random'
        self.state = state

        #hyperparams
        self.ep=ep
        self.sgd= sgd
        self.wd=wd
        self.bs=bs
        self.ls_magnitude=ls_magnitude
        self.amp = amp
        self.llw=llw
        self.filename = filename
        self.kernel_size = kernel_size              #used for nips22 review larger kernel
        self.model_name = model_name



        if dataset=='pmnist':
            self.folder = '/home/diego/ricerca/trained_models/models/pmnist/original'
        if dataset=='pmnist_wd':
            self.folder = '/home/diego/ricerca/trained_models/models/pmnist/regularized'
        if dataset.startswith('cifar'):
            self.folder = f'/home/diego/Documents/dottorato/ricerca/trained_models/models/{dataset}/{self.net_name}'
            # if ablation:
            #     self.folder += f'/ablation'
        if dataset == 'normal':
            self.folder = 'dummy'

        if model_path is not None:
            self.folder = model_path





        if root_results is not None:
            self.root_results = root_results
        else:
            self.root_results = './results'

        self.results_features = f'./results/{self.data_name}/{self.net_name}/features'
        if not os.path.isdir(self.results_features) and dataset!= 'imagenet':
            pathlib.Path(self.results_features).mkdir(parents=True, exist_ok=True)

        if dataset=='imagenet':
            self.results_features = '/home/diego/Documents/dottorato/ricerca/present/net_mitosis/analysis/v2/results/imagenet/features'

        self.results_acc = f'{self.root_results}/{self.data_name}/{self.net_name}/accuracy'
        if not os.path.isdir(self.results_acc):
            pathlib.Path(self.results_acc).mkdir(parents=True, exist_ok=True)

        self.results_loss = f'{self.root_results}/{self.data_name}/{self.net_name}/loss'
        if not os.path.isdir(self.results_loss):
            pathlib.Path(self.results_loss).mkdir(parents=True, exist_ok=True)

        self.results_ensamble = f'{self.root_results}/{self.data_name}/{self.net_name}/ensamble'
        if not os.path.isdir(self.results_ensamble):
            pathlib.Path(self.results_ensamble).mkdir(parents=True, exist_ok=True)

        self.results_r2 = f'{self.root_results}/{self.data_name}/{self.net_name}/r2'
        if not os.path.isdir(self.results_r2):
            pathlib.Path(self.results_r2).mkdir(parents=True, exist_ok=True)

        self.results_ids = f'{self.root_results}/{self.data_name}/{self.net_name}/id'
        if not os.path.isdir(self.results_ids):
            pathlib.Path(self.results_ids).mkdir(parents=True, exist_ok=True)

    def compute_accuracy(self, n_trials, width_tot, seed=0, chunk_size_ = None, cpt = None, verbose = True):
        print(f'computing accuracy {self.net_name}_{width_tot},\n  seed = {seed}, weight_decay = {self.wd}')
        print(f'dataset = {self.data_name}, train = {self.trainset}' )

        if self.data_name == 'imagenet':
            self.features, self.weights, self.bias = self._get_params_imagenet(width_tot = width_tot, seed = seed, cpt= cpt)
        else:
            self.features, self.weights, self.bias =  self._get_params(width_tot = width_tot, seed = seed,
                                                                cpt = cpt,verbose = verbose)

        chunk_size = np.sort(width_tot//np.array([2**i for i in range( int(math.log2(width_tot)) +1 )]))
        if self.llw:
            chunk_size = [2**i for i in range( int(math.log2(width_tot*self.llw)) +1 )]

        if chunk_size_ is not None: chunk_size = chunk_size_
        print(f'chunk_sizes = {chunk_size}')

        criterion, targets = self._get_loss_targets(width_tot=width_tot, seed=seed)
        label = self._get_label(cpt = cpt, seed = seed)

        for w in chunk_size:
            print(f'chunk_size = {w}')
            acc_av, loss_av = self._compute_accuracy_chunk(n_trials, width_tot, w, self.features, self.weights, self.bias, targets, criterion)
            print(f'acc_av = {np.mean(acc_av): .4f}, loss_av = {np.mean(loss_av): .4f}')

            if self.trainset:
                np.save(f'{self.results_acc}/acc_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_train', acc_av)
                np.save(f'{self.results_loss}/loss_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_train', loss_av)
            else:
                print('saving...')
                np.save(f'{self.results_acc}/acc_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_test', acc_av)
                np.save(f'{self.results_loss}/loss_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_test', loss_av)


    def compute_r2(self, n_trials,  width_tot, seed = 0, chunk_size_ = None, cpt = None, nimg_class = None, save = True, compute_sym = False, norm = 'mean'):

        print(f'computing r2 for {self.data_name} on {self.net_name}_{width_tot},  seed = {seed}\ncomputation is done on the TEST set')
        self.trainset = False

        if self.data_name == 'imagenet':
            features, _, _= self._get_params_imagenet(width_tot = width_tot, seed = seed, cpt= cpt)
        else:
            features, _, _, =  self._get_params(width_tot = width_tot, seed = seed, cpt= cpt)

        chunk_size = np.sort(int(width_tot)//np.array([2**i for i in range( int(math.log2(width_tot)) +1 )]))
        if chunk_size_ is not None:
            chunk_size = chunk_size_


        print(f'chunk_sizes = {chunk_size}')
        if self.data_name != 'normal':
            subset = self._reduce_indices(nimg_class)
            features = self.features[np.array(subset)]

        ndata = len(features)

        label = self._get_label(cpt = cpt, seed = seed)
        if compute_sym:
            label = label+'_sym'

        for w in chunk_size:
            print(f'chunk_size = {w}' )
            r2, mean_corr = self._compute_r2_chunk(n_trials, features, w, compute_sym= compute_sym, norm = norm)
            print(f'r2_global = {np.mean(r2): .4f}, mean_corr = {np.mean(mean_corr): .4f}')
            if save:
                np.save(f'{self.results_r2}/r2_global_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_ndata{ndata//1000}k_test', r2)
                if norm=='fro':
                    np.save(f'{self.results_r2}/mean_corr_fro_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_ndata{ndata//1000}k_test', mean_corr)
                else:
                    np.save(f'{self.results_r2}/mean_corr_{self.net_name}_{width_tot}_{label}_w{w}_nrep{n_trials}_ndata{ndata//1000}k_test', mean_corr)

    def compute_ensamble_average_rep(self, w, seeds, nrep, sizes, verbose = False):

        if not isinstance(nrep, list):
            nrep = [nrep for i in range(len(sizes))]
            flaglist = False
        else:
            flaglist=True

        self.trainset = False
        print(f'net = {self.net_name}_{w}; ensamble size = {len(seeds)}')

        if self.data_name=='imagenet':
            targets = torch.arange(1000).repeat(50, 1).T.reshape(-1)
            for i in range(len(seeds)):
                actual_targets = torch.load(f'{self.results_features}/targets_{self.net_name}_{w}_ep{self.ep}_seed{seeds[i]}_test.pt')
                assert torch.all(targets==actual_targets)
            targets = targets.type(torch.long)
            #criterion, targets = self._get_loss_targets(width_tot=w, seed=seeds[0])
        else:
            criterion, targets = self._get_loss_targets(width_tot=w, seed=seeds[0])
        seeds_tmp = self._get_seeds_ens(seeds, sizes, nrep)

        acc_ens = []
        for i in range(len(sizes)):
            print(f'computing subsize {sizes[i]}, nsamples = {len(seeds_tmp[i])}')
            acc_ens_tmp = []
            for j in range(len(seeds_tmp[i])):
                if verbose: print(f'rep {j}')

                predictions = self._get_ens_pred(seeds_tmp[i][j], w,  verbose)
                accuracy = self._compute_accuracy(predictions, targets)
                acc_ens_tmp.append(accuracy)

            print(np.mean(acc_ens_tmp))
            acc_ens.append(acc_ens_tmp)

        label = self._get_label()
        if flaglist:
            for i in range(len(acc_ens)):
                np.save(f'{self.results_ensamble}/ensamble_averages_{self.net_name}_{w}_{label}_n{len(seeds)}_{sizes[i]}_balanced.npy', np.array(acc_ens[i]))
        else:
            for i in range(len(acc_ens)):
                np.save(f'{self.results_ensamble}/ensamble_averages_{self.net_name}_{w}_{label}_n{len(seeds)}_{sizes[i]}_nrep{nrep[0]}.npy', np.array(acc_ens[i]))

    def compute_ensamble_average(self, w, seeds, wd=None, verbose = True):

        self.trainset = False
        print(f'net = {self.net_name}; ensamble size = {len(seeds)}')

        criterion, targets = self._get_loss_targets()
        predictions = self._get_ens_pred(seeds, w, verbose)
        acc_ens = self._compute_accuracy(predictions, targets)
        print(f'accuracy = {acc_ens}')

        label = self._get_label()
        np.save(f'{self.results_ensamble}/ensamble_averages_{self.net_name}_{w}_{label}_n{len(seeds)}.npy', acc_ens)

#*******************************************************************************
    def _get_label(self, cpt = None, seed = None):

        if self.data_name=='pmnist' or self.data_name=='pmnist_wd':
            assert self.bs is not None
            assert self.wd is not None

            label = f'wd{self.wd}_bs{self.bs}'
            if self.batch_norm:
                label = f'batch_norm_wd{self.wd}_bs{self.bs}'

        elif self.data_name=='cifar10' or self.data_name=='cifar100' or self.data_name == 'imagenet':
            if self.state=='random':
                label = ''
            else:
                assert self.ep is not None
                label = f'ep{self.ep}'

            if self.wd is not None:
                label = f'ep{self.ep}_wd{self.wd}'
        elif self.data_name=='normal':
            label=f'gaussian_inputs'

        if cpt is not None:
            label += f'_cpt{cpt}'

        if self.label_smoothing:
            label += '_label_smoothing'
            if self.ls_magnitude is not None:
                label += f'{self.ls_magnitude}'

        if seed is not None:
            label += f'_seed{seed}'

        if self.llw:
            label+=f'_llw{self.llw}'

        if self.state=='random' or self.data_name=='normal':
            label+=f'_random_weights'


        if self.filename is not None:
            label+=f'_{self.filename}'

        return label

    def _get_loss_targets(self,width_tot=None, seed=None):

        if self.data_name == 'cifar10' or self.data_name == 'cifar100':
            criterion = nn.CrossEntropyLoss()
            targets = self.load_data_cifar(self.trainset)[3]
        elif self.data_name == 'imagenet':
            criterion = nn.CrossEntropyLoss()
            targets = self.load_targets_imagenet(self.trainset, width_tot, seed)

        elif self.data_name == 'pmnist'or self.data_name == 'pmnist_wd':
            criterion = square_hinge()
            targets = self.load_data_pmnist(self.trainset)[3]

        return criterion, targets

    def _get_seeds_ens(self,seeds, sizes, nrep):

        seeds_tmp = []
        for i in range(len(sizes)):
            if sizes[i]==len(seeds):
                seeds_tmp.append(np.random.choice(seeds, len(seeds), replace = False).reshape(1, -1))
            elif sizes[i]==1:
                seeds_tmp.append(np.random.choice(seeds, len(seeds), replace = False).reshape(-1, 1))
            else:
                seeds_tmp.append(np.empty((nrep[i], sizes[i]), dtype = 'int') )
                for j in range(nrep[i]):
                    seeds_tmp[i][j] = np.random.choice(seeds, sizes[i], replace = False)

        return seeds_tmp

    def _get_ens_pred(self, seeds, w, verbose):
        for k, seed in enumerate(seeds):
            if self.data_name=='imagenet':
                predictions_tmp = torch.load(f'{self.results_features}/logits_{self.net_name}_{w}_seed{seed}_test.pt')
                if k == 0:
                    predictions = predictions_tmp
                else:
                    predictions += predictions_tmp
            else:
                features, weights, bias =  self._get_params(width_tot= w, seed = seed, verbose = verbose)
                if k == 0:
                    predictions = torch.matmul(features, weights.T)+bias
                else:
                    predictions += torch.matmul(features, weights.T)+bias
        return predictions

    def _reduce_indices(self, nimg_class):


        if self.data_name == 'pmnist_wd':
            nimg_cl = 5000
        if self.data_name == 'cifar10':
            nimg_cl = 1000
        if self.data_name == 'cifar100':
            nimg_cl = 100
        if self.data_name == 'imagenet':
            nimg_cl = 50

        if nimg_class is not None:
            nimg_cl = nimg_class
            print(f'keeping nimg_cl equal to {nimg_cl}')
        else:
            print('keeping nimg equal to |test set|')


        if self.data_name == 'imagenet':
            targets = torch.load(f'{self.results_features}/targets_resnet50_8192_ep120_seed2_test.pt')
            targets = targets.numpy()
        else:
            dataset, loader, batch_size, phantom = self._get_data()

            if isinstance(dataset.targets, list):
                targets = np.array(dataset.targets)
            else:
                targets = dataset.targets.numpy()

        categories = list(set(targets))
        subset = []
        for t in categories:
            ind = np.where(targets==t)[0]
            subset.extend(list(np.random.choice(ind, nimg_cl, replace=False)))

        return subset

    def _compute_accuracy_chunk(self, n_trials, W, w, features, weights, bias, targets, criterion):
        print(f'w = {w}')
        acc_av = np.empty(n_trials)
        loss_av = np.empty(n_trials)

        if self.llw: W = int(self.llw)

        for i in range(n_trials):
            ind = np.random.choice(W, size = w, replace = False)
            outputs = torch.matmul(features[:, ind], weights[:, ind].T)+bias

            loss_av[i] = criterion(outputs, targets).item()

            accuracy = self._compute_accuracy(outputs, targets)

            acc_av[i] = accuracy

        return acc_av, loss_av

    def _compute_accuracy(self, predictions, targets):

        if self.data_name=='cifar10' or self.data_name=='cifar100' or self.data_name=='imagenet':
            _, max_logit = predictions.max(1)
            accuracy = max_logit.eq(targets).sum().item()/len(targets)
        elif self.data_name=='pmnist'or self.data_name=='pmnist_wd':
            accuracy = 1.*torch.sum(predictions.reshape(-1)*targets>0)/targets.shape[0]

        return accuracy

    def _compute_id_chunk(self, width_tot, chunk_size, features, n_trials, algo='r2n'):

        assert algo=='r2n' or algo=='2nn'

        ids = np.empty(n_trials)
        err = np.empty(n_trials)
        id2NN = np.empty(n_trials)

        for i in range(n_trials):
            index = np.random.permutation(width_tot)
            act_tmp = features[:, index[:chunk_size]]
            d = Data(act_tmp.numpy(), verbose = True)
            if algo=='2nn':
                id2NN_tmp, err2NN_tmp, rs2NN = d.compute_id_2NN(return_id=True, algorithm = 'base')
            else:
                ids_tmp, err_tmp, rs = d.return_id_scaling_r2n()
            print(f'nrep = {i}, w = {chunk_size}; idr2n =  {ids_tmp[0]}')
            ids[i] = ids_tmp[0]
            err[i] = err_tmp[0]
            #id2NN[i] = id2NN_tmp
        return ids, err

    def _compute_r2_chunk(self, n_trials, features, w, compute_sym=False, norm = 'mean'):
        features = features.numpy()
        print(f'compute r2 w = {w}')
        r2_global = np.empty(n_trials)
        mean_corr = np.empty(n_trials)

        r2_sym = np.empty(n_trials)
        mean_corr_sym = np.empty(n_trials)

        nfeatures = features.shape[1]
        chunk_size = w

        for i in range(n_trials):
            ind_tot = np.arange(nfeatures)
            perm = np.random.permutation(np.arange(nfeatures))
            ind1 = perm[:chunk_size]
            ch1 = features[:, ind1]

            if not compute_sym:
                reg = Ridge(1e-8).fit(ch1, features)
                r2_global[i] = r2_score(features, np.matmul(ch1, reg.coef_.T)+reg.intercept_, multioutput = 'variance_weighted')

                #mean correlation
                res = features - reg.predict(ch1) # N x n_feat
                mean_corr[i] = self._compute_mean_corr(res, reg = 1e-8, mode = norm)

            if compute_sym and w < (nfeatures+1)/2:
                ind2 = perm[chunk_size:2*chunk_size]
                ch2 = features[:, ind2]

                reg1 = Ridge(1e-8).fit(ch1, ch2)
                reg2 = Ridge(1e-8).fit(ch2, ch1)

                r2_1 = r2_score(ch2, np.matmul(ch1, reg1.coef_.T)+reg1.intercept_, multioutput = 'variance_weighted')
                r1_2 = r2_score(ch1, np.matmul(ch2, reg2.coef_.T)+reg2.intercept_, multioutput = 'variance_weighted')
                r2_sym[i] = 0.5*(r2_1 + r1_2)

                res2_1 = ch2 - reg1.predict(ch1) # N x n_feat

                mean_corr2_1 = self._compute_mean_corr(res2_1,  reg = 1e-8, mode=norm)

                res1_2 = ch2 - reg1.predict(ch1) # N x n_feat
                mean_corr1_2 = self._compute_mean_corr(res1_2,  reg = 1e-8, mode=norm)

                mean_corr_sym[i] = 0.5*(mean_corr2_1+mean_corr1_2)

        if compute_sym:
            return r2_sym, mean_corr_sym
        else:
            return r2_global, mean_corr

    def _compute_mean_corr(self, res, reg = 1e-8, mode = 'mean'):

        cov = np.cov(res.T) # n_feat x n_feat

        #normalization (regularized)
        norm = np.diag(cov) # n_feat
        norm = (np.array([norm]).T*norm)**0.5 + reg # n_feat x n_feat
        #corr
        corr = np.divide(cov, norm)
        #average upper triugular matrix
        nondiag_entries = corr[np.tril_indices_from(corr, k = -1)]


        if mode=='mean':
            corr = np.mean(abs(nondiag_entries))
        elif mode=='fro':
            corr = np.linalg.norm(nondiag_entries)/len(nondiag_entries)

        return corr

    def _estract_features(self, w, seed, nlay = 6, sched ='CosineAnnealingLR', cpt = None):
        with torch.no_grad():
            print(f'estracting features {self.net_name}_{w},  seed = {seed}' )

            if self.data_name=='normal':
                dataset, loader, batch_size, phantom = self._get_data_random(ndata=10000, shape=(3, 32, 32), seed = seed)
                self.state = 'random'
                model, modules, self.weights, self.bias, names = self._get_model(w, seed, nlay = nlay, sched = sched, cpt=cpt)
            else:
                dataset, loader, batch_size, phantom = self._get_data()
                model, modules, self.weights, self.bias, names = self._get_model(w, seed, nlay = nlay, sched = sched, cpt=cpt)

            print(names)
            model.to(self.device)
            model.eval()

            del modules[-3]

            label = self._get_label(cpt = cpt, seed = seed)
            torch.save(self.weights, f'{self.results_features}/weights_{self.net_name}_{w}_{label}.pt')
            torch.save(self.bias, f'{self.results_features}/bias_{self.net_name}_{w}_{label}.pt')

            embdims = activations_utils.get_layer_depths(phantom, model, modules, self.device)
            nsamples = len(dataset)

            for l, module in enumerate(modules):
                if l==len(modules)-2:
                    out = torch.empty(nsamples, embdims[l])
                    if self.net_name.startswith('densenet'):
                        out = torch.empty(nsamples, embdims[l]//64)
                        out = activations_utils.get_activations(loader, model, self.device, out, batch_size, nsamples, module, postproc=True)
                    else:
                        out = activations_utils.get_activations(loader, model, self.device, out, batch_size, nsamples, module)
                    self.features = out.to('cpu')
                    if self.trainset:
                        torch.save(self.features, f'{self.results_features}/features_{self.net_name}_{w}_{label}_train.pt')
                    else:
                        torch.save(self.features, f'{self.results_features}/features_{self.net_name}_{w}_{label}_test.pt')

    def _get_params(self, width_tot, seed, cpt=None, verbose=False):

        label = self._get_label(cpt = cpt, seed = seed)


        if self.recompute_features:
            self._estract_features(width_tot, seed, cpt=cpt)

        elif self.trainset and not self.recompute_features:
            try:
                self.features = torch.load(f'{self.results_features}/features_{self.net_name}_{width_tot}_{label}_train.pt')
                self.weights = torch.load(f'{self.results_features}/weights_{self.net_name}_{width_tot}_{label}.pt')
                self.bias = torch.load(f'{self.results_features}/bias_{self.net_name}_{width_tot}_{label}.pt')
                if verbose:
                    print('features already computed')
            except:
                self._estract_features(width_tot, seed, cpt=cpt)
        elif not self.recompute_features:
            try:
                self.features = torch.load(f'{self.results_features}/features_{self.net_name}_{width_tot}_{label}_test.pt')
                self.weights = torch.load(f'{self.results_features}/weights_{self.net_name}_{width_tot}_{label}.pt')
                self.bias = torch.load(f'{self.results_features}/bias_{self.net_name}_{width_tot}_{label}.pt')
                if verbose:
                    print('features already computed')
            except Exception as e:
                if self.data_name == 'imagenet':
                    print(e)
                else:
                    self._estract_features(width_tot, seed, cpt=cpt)

        return self.features, self.weights, self.bias

    def _get_params_imagenet(self, width_tot, seed, cpt=None, verbose=True):

        label = self._get_label(cpt = cpt, seed = seed)

        self.features = torch.load(f'{self.results_features}/features_{self.net_name}_{width_tot}_{label}_test.pt')
        self.weights = torch.load(f'{self.results_features}/weights_{self.net_name}_{width_tot}_{label}.pt')
        self.bias = torch.load(f'{self.results_features}/bias_{self.net_name}_{width_tot}_{label}.pt')

        return self.features, self.weights, self.bias

    def _get_model(self, w, seed, nlay, sched, cpt = None):

        torch.manual_seed(seed)


        if self.data_name=='cifar10' or self.data_name=='cifar100' or self.data_name=='normal':

            if w == 16 and self.net_name=='wide_resnet28':
                net_name_tmp = self.net_name+'_1_inpl4'
                inplanes = 4
                w = 1

            elif w == 32 and self.net_name=='wide_resnet28':
                net_name_tmp = self.net_name+'_1_inpl8'
                inplanes = 8
                w = 1

            elif self.net_name == 'wide_resnet28':
                net_name_tmp = self.net_name+f'_{w//64}'
                inplanes = 16
                w = w//64
            elif self.net_name.startswith('densenet28'):
                net_name_tmp = self.net_name+f'_{int(w/3.75)}'
                inplanes = int(w/3.75)
            elif self.net_name.startswith('densenet40'):
                net_name_tmp = self.net_name+f'_{int(w/5.5)}'
                inplanes = int(w/5.5)

            if self.data_name == 'cifar10' or self.data_name =='normal': nc=10
            elif self.data_name == 'cifar100': nc=100

            if self.net_name =='wide_resnet28':
                model = wide_resnet28(num_classes=nc,
                                    widening_factor = w,
                                    inplanes = inplanes,
                                    #last_layer_width = self.llw, #only used is a review not implemented for nips22
                                    kernel_size = self.kernel_size)
            elif self.net_name=='densenet28':
                model = densenet28(num_classes=nc,num_init_features=inplanes, growth_rate = inplanes//2)
            elif self.net_name=='densenet40':
                model = densenet40(num_classes=nc,
                                num_init_features=inplanes,
                                growth_rate = inplanes//2,
                                #out_features = self.llw,       #only used is a review not implemented for nips22
                                kernel_size = self.kernel_size)

                if self.filename=='stem_type_imagenet':
                    model = densenet40(num_classes=nc,num_init_features=inplanes, growth_rate = inplanes//2, stem_type='imagenet')


            if self.amp:
                filename = f'{self.data_name}_{net_name_tmp}_ep200_seed{seed}_wd{self.wd}_ls{self.ls_magnitude}_amp'
            else:
                #filename = f'{self.data_name}_{net_name_tmp}_ep200_optSGD_sched{sched}_seed{seed}'
                filename = f'{self.data_name}_{net_name_tmp}_ep200_seed{seed}'

            if not self.amp:
                #if self.label_smoothing:
                #    filename += f'_label_smoothing'
                #    print('label_smoothing')
                if self.wd is not None:
                    filename+=f'_wd{self.wd}'
                if self.ls_magnitude is not None:
                    filename+=f'_ls{self.ls_magnitude}'
                if self.llw:
                   filename = filename+f'_llw{self.llw}'

            if self.filename is not None:
                filename = filename+f'_{self.filename}'

            if cpt is not None:
                path = f'{self.folder}/{filename}_model_{cpt}_cpt.pth.tar'
            else:
                path = f'{self.folder}/{filename}_model_best.pth.tar'

            if self.model_name is not None:
                path = f'{self.folder}/{self.model_name}'

            if self.state !='random':
                dict = torch.load(path, map_location = 'cpu')
                model.load_state_dict(dict['state_dict'])
            else:
                print('loading untrained model')

            if self.net_name=='wide_resnet28':
                weights = model.state_dict()['fc.weight'].to('cpu')
                bias = model.state_dict()['fc.bias'].to('cpu')
            elif self.net_name.startswith('densenet'):
                weights = model.state_dict()['classifier.weight'].to('cpu')
                bias = model.state_dict()['classifier.bias'].to('cpu')
            if self.net_name =='wide_resnet28':
                modules, names, depths = layer_loader.getWideResNetDepths(model, red = 10)
            elif self.net_name.startswith('densenet'):
                modules, names, depths = layer_loader.getdensenet(model, red = 10)


        elif self.data_name=='pmnist':
            model = fully_connected_old(width = w, depth = nlay)
            path =  f'{self.folder}/W{w}_seed{seed}_nlay{nlay}_best.pth.tar'

            dict = torch.load(path)
            model.load_state_dict(dict['state_dict'])
            weights = model.state_dict()['linear_out.weight'].to('cpu')
            bias = model.state_dict()['linear_out.bias'].to('cpu')

            modules, names, depths = layer_loader.get_fully_connected_biroli_depths(model)

        elif self.data_name=='pmnist_wd':
            model = fully_connected_new(w, depth = nlay, batch_norm = True, dropout = False)
            if cpt is not None:
                path =  f'{self.folder}/W{w}_seed{seed}_nlay6_wd{self.wd}_lr0.001_batch_norm_bs{self.bs}_sgd_{cpt}_cpt.pth.tar'
            else:
                path =  f'{self.folder}/W{w}_seed{seed}_nlay6_wd{self.wd}_lr0.001_batch_norm_bs{self.bs}_sgd_best.pth.tar'
            dict = torch.load(path)
            model.load_state_dict(dict['state_dict'])
            weights = model.state_dict()['linear_out.weight'].to('cpu')
            bias = model.state_dict()['linear_out.bias'].to('cpu')

            modules, names, depths = layer_loader.get_fully_connected_biroli_depths(model)

        return model, modules, weights, bias, names

    def _get_data_random(self, ndata, shape, seed):

        phantom = torch.zeros(1, 3, 32, 32)
        batch_size = 4
        dataset = NormalDataset(ndata=ndata, shape =shape, seed = seed)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=10, pin_memory=True)

        return dataset, loader, batch_size, phantom

    def _get_data(self,):

        if self.data_name == 'cifar10' or self.data_name == 'cifar100':
            phantom = torch.zeros(1, 3, 32, 32)
            dataset, loader, batch_size, _ = self.load_data_cifar(self.trainset)

        elif self.data_name == 'pmnist' or self.data_name=='pmnist_wd':
            phantom = torch.zeros(1, 10)
            dataset, loader, batch_size, _ = self.load_data_pmnist(self.trainset)

        return dataset, loader, batch_size, phantom

    def load_data_cifar(self, train):

        if self.data_name=='cifar10':
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            data_path = '/home/diego/ricerca/datasets/cifar10'
            dataset = datasets.CIFAR10(f'{data_path}', train = train, transform = transform_val)

            targets = torch.tensor(dataset.targets)
            targets = targets.type(torch.long)

        elif self.data_name=='cifar100':
            transform_val = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
            data_path = '/home/diego/ricerca/datasets/cifar100'
            dataset = datasets.CIFAR100(f'{data_path}', train = train, transform = transform_val)

            # if self.supercl ==True:
            #     sup_tg = ut.c100super(dataset)
            #     tg = []
            #     for i in range(len(dataset.targets)):
            #         tg.append(np.where(sup_tg==dataset.targets[i])[0][0])
            #     targets = torch.tensor(tg)
            #     targets = targets.type(torch.long)
            # else:
            targets = torch.tensor(dataset.targets)
            targets = targets.type(torch.long)

        batch_size = 4
        # subset = self._reduce_indices()
        # if self.reduce:
        #     dataset = dataset[]

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=10, pin_memory=True)

        # if self.reduce:
        #     targets = targets[subset]

        return dataset, loader, batch_size, targets

    def load_targets_imagenet(self, train, width_tot, seed):

        if self.trainset:
            pass
        else:
            targets = torch.arange(1000).repeat(50, 1).T.reshape(-1)
            actual_targets = torch.load(f'{self.results_features}/targets_{self.net_name}_{width_tot}_ep{self.ep}_seed{seed}_test.pt')
            assert torch.all(targets==actual_targets)
            targets = targets.type(torch.long)

        return targets

    def load_data_pmnist(self, train):
        data_path = '/home/diego/ricerca/datasets/mnist/MNIST_pc/std_not_rescaled/10pc'
        dataset = my_dataset.MNIST_pc(f'{data_path}', train=not train)
        batch_size = len(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=10, pin_memory=True)

        targets = dataset.targets
        #targets = targets.type(torch.long)
        return dataset, loader, batch_size, targets

class square_hinge:
    def __init__(self,):
        a=1
        # self.outputs = outputs
        # self.targets = targets

    def __call__(self, outputs, targets):
        batch_size = len(targets)
        delta = 1 - outputs*targets.view(batch_size, 1)
        delta[delta<0]=0
        delta = delta**2
        loss = 0.5*torch.sum(delta)/batch_size
        return loss
