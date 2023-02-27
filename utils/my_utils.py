import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import sys
from torchvision import models
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

#*******************************************************************************
#cifar100 superclasses
superclass = np.array([['beaver', 'dolphin', 'otter', 'seal', 'whale'],
              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
              ['bottle', 'bowl', 'can', 'cup', 'plate'],
              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
              ['bed', 'chair', 'couch', 'table', 'wardrobe'],
              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
              ['bridge', 'castle', 'house', 'road', 'skyscraper'],
              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
              ['crab', 'lobster', 'snail', 'spider', 'worm'],
              ['baby', 'boy', 'girl', 'man', 'woman'],
              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']])

def c100super(dataset, superclass=superclass):
    sup_tg = np.empty((20, 5), dtype = 'int')
    for i in range(len(superclass)):
        for j in range(len(superclass[i])):
            sup_tg[i, j] = dataset.class_to_idx[superclass[i][j]]
    return sup_tg

#-------------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def reduce(dataset, nimg_cl, ncl_tot = None, nclasses=None, class_labels = None, randomize = False):

    index_tmp = []
    if ncl_tot is not None:
        class_labels = np.random.choice(ncl_tot, nclasses, replace = False)
    elif class_labels is None:
        class_labels = np.arange(nclasses)
    else:
        class_labes = np.array(class_labels)
    for i in class_labels:
        l = (np.array(dataset.targets)==i).nonzero()
        if randomize:
            indices = random.sample(list(l[0]), nimg_cl)
        else:
            indices = list(l[0][:nimg_cl])
        index_tmp.append(indices)

    index = [i for sublist in index_tmp for i in sublist]

    return torch.utils.data.Subset(dataset, index)

def get_search_subsets(dataset, nimg_cl_tr, nimg_cl_val, nclasses=None):

    index_tmp_tr = []
    index_tmp_val = []
    class_labels = np.arange(nclasses)

    for i in class_labels:
        l = np.where(np.array(dataset.targets)==i)[0]
        l = np.random.permutation(l)
        ind_train = l[:nimg_cl_tr]
        ind_val = l[nimg_cl_tr: nimg_cl_tr+nimg_cl_val]

        index_tmp_tr.append(ind_train)
        index_tmp_val.append(ind_val)

    index_tr = [i for sublist in index_tmp_tr for i in sublist]
    index_val = [i for sublist in index_tmp_val for i in sublist]

    trainset = torch.utils.data.Subset(dataset, index_tr)
    valset = torch.utils.data.Subset(dataset, index_val)

    return trainset, valset

def resume(path, model, optimizer, scheduler):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")
        return model, optimizer, scheduler, best_acc1, start_epoch
    else:
        print(f"=> no checkpoint found at '{path}'")

#-------------------------------------------------------------------------------
def get_gradient_norm(optimizer):
    norm = 0
    for p in optimizer.param_groups[0]['params']:

        if p.grad is None:
            continue
        norm+= torch.sum(p.grad**2).item()
    return norm**0.5
#-------------------------------------------------------------------------------

def save_checkpoint(state, is_best, folder, filename = '/model', register_epoch = None):
    if not os.path.isdir(folder): pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    if register_epoch is None: path = folder + filename
    else:
        path = folder + filename + f'_{register_epoch}'
    torch.save(state, path+'_cpt.pth.tar')
    if is_best: shutil.copyfile(path+'_cpt.pth.tar', path+'_best.pth.tar')

def write_params(model=None, optimizer=None, transforms=None, epochs=None, batch_size=None, folder = None, filename = f'/params', scheduler = None):

    if not os.path.isdir(folder): pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    with open(folder+filename, 'w') as f:

        f.write(f'model: {model}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')

        f.write(f'optimizer: {optimizer.__class__.__name__}\n')
        for key, value in optimizer.state_dict()['param_groups'][0].items():
            if key != 'params':
                f.write(f'\t{key}: {value}\n')

        if scheduler is not None:
            f.write(f'scheduler: {scheduler.__class__.__name__}\n')
            for key, value in scheduler.state_dict().items():
                f.write(f'\t{key}: {value}\n')
        if transforms is not None:
            f.write(f'transforms\n')
            for tr in transforms:
                f.write(f'\t{tr}\n')

def write_stats(epoch = None, val_acc=None, tr_acc=None, val_loss=None, tr_loss=None, aux_loss = None,
                    cross_entropy = None, gradient = None, folder=None, filename = '/stats', start=False):

    if not os.path.isdir(f'{folder}'): pathlib.Path(f'{folder}').mkdir(parents=True, exist_ok=True)

    if start:
        action = 'w'
        with open(folder+filename, action) as f:
            f.write('epoch test_acc train_acc test_loss train_loss\n')
    else:
        action = 'a'
        with open(folder+filename, action) as f:
            if aux_loss is not None:
                if gradient is not None:
                    f.write(f'{epoch} {val_acc: .7f} {tr_acc: .7f} {val_loss: .7f} {tr_loss: .7f} {aux_loss: .7f} {cross_entropy: .7f} {gradient: .7f}\n')
                else:
                    f.write(f'{epoch} {val_acc: .7f} {tr_acc: .7f} {val_loss: .7f} {tr_loss: .7f} {aux_loss: .7f} {cross_entropy: .7f}\n')
            elif gradient is not None:
                f.write(f'{epoch} {val_acc: .7f} {tr_acc: .7f} {val_loss: .7f} {tr_loss: .7f} {gradient}\n')
            else:
                f.write(f'{epoch} {val_acc: .7f} {tr_acc: .7f} {val_loss: .7f} {tr_loss: .7f}\n')

def write_ids(epoch=None, top1=None, ids=None, layers=None, folder=None, filename = '/ids', start = False):
    if not os.path.isdir(f'{folder}'): pathlib.Path(f'{folder}').mkdir(parents=True, exist_ok=True)

    if start: action = 'w'
    else: action = 'a'

    with open(folder+filename, action) as f:
        if action == 'w': f.write(f'epoch   top1   layers\n')
        else:
            f.write(f'{epoch:5d} {top1:7.2f}')
            for id in ids: f.write(f'{id:6.2f}')
            f.write('\n')

def write_results(epoch, acc, time=None, folder=None, filename = '/results'):
    if not os.path.isdir(f'{folder}'): pathlib.Path(f'{folder}').mkdir(parents=True, exist_ok=True)

    with open(folder+filename, 'w') as f:
        if time is not None:
            f.write('epoch best_acc1 time(min) \n')
            f.write(f'{epoch} {acc} {time/60}')
        else:
            f.write('epoch best_acc1 \n')
            f.write(f'{epoch} {acc}')
