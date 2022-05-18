import argparse
import random
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from utils.randaugment import RandAugment
from utils.wide_resnet import wide_resnet28, wide_resnet16, wide_resnet10, wide_resnet40
import utils.my_utils as ut
#*******************************************************************************
def train(loader, model, criterion, optimizer, epoch = None, scheduler = None):

    model.train()
    tr_loss = 0
    correct = 0
    total = 0
    iters = len(loader)

    for i, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler.__class__.__name__=='CosineAnnealingWarmRestarts' or scheduler.__class__.__name__=='CosineAnnealingLR':
            scheduler.step(epoch+i/iters)

        tr_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        total += targets.size(0)

    return 1.0*correct/total, tr_loss/total

def validate(loader, model, criterion):

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 1.0*correct/total, val_loss/total

def get_transforms(norm_mean, norm_std, randaugment, n, m, cutout_size = 8,):
    trans = [transforms.RandomCrop(32, padding=4, padding_mode ='reflect'), transforms.RandomHorizontalFlip(),]
    if randaugment:
        trans += [RandAugment(n, m, cutout_size = cutout_size)]

    trans+=[transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std) ]
    transform_tr = transforms.Compose(trans)

    transform_val = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(norm_mean, norm_std)])

    return transform_tr, transform_val

#*******************************************************************************
parser = argparse.ArgumentParser()
"model specs"
parser.add_argument('--net_name', default = 'wide_resnet28',)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--inplanes', default=16, type=int)
parser.add_argument('--widening_factor', default=2, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--resume', action='store_true')

"optimization"
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float, )
parser.add_argument('--label_smoothing', action='store_true')
parser.add_argument('--ls_magnitude', default=0.1, type=float)
parser.add_argument('--pdrop', default=None, type=float)
parser.add_argument('--search', action='store_true')

"dataset and augmentation params"
parser.add_argument('--data', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                    help='dataset choice')
parser.add_argument('--data_path', type=str,  default = '.',
                    help='dataset path')
parser.add_argument('--randaugment', action='store_true',
                    help='whether to augment the data with randaugment (default false)')
parser.add_argument('--n', default=1, type=int,
                    help='randaugment param')
parser.add_argument('--m', default=5, type=int,
                    help='randaugment param')
parser.add_argument('--cutout', action='store_true',
                    help='whether to use cutout')
parser.add_argument('--superclasses', action='store_true',
                    help='whether to use superclasses (c100 only)')
parser.add_argument('--nimg_cl_tr', default = None, type = int)
parser.add_argument('--nimg_cl_val', default = None, type = int)


parser.add_argument('--results_path', type=str, default = './models',
                    help='where to save the model')
parser.add_argument('--filename', type=str,
                    help='optional filename description')
parser.add_argument('--resume_folder', type=str,  default = None,
                    help='folder where previously trained model is stored')
parser.add_argument('--resume_filename', type=str,  default = None,
                    help='name of previously trained model')

args = parser.parse_args()
#*******************************************************************************
if args.seed is not None:
   random.seed(args.seed)
   torch.manual_seed(args.seed)

#*******************************************************************************
"Data loading code"
if args.data == 'cifar100':
    args.data_path = args.data_path+'/cifar100'
    norm_mean = (0.5071, 0.4865, 0.4409)
    norm_std = (0.2673, 0.2564, 0.2762)
elif args.data == 'cifar10':
    args.data_path = args.data_path+'/cifar10'
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.247, 0.243, 0.261)

"initialize data trainsforms"
transform_tr, transform_val = get_transforms(norm_mean, norm_std, randaugment = args.randaugment, n = args.n, m=args.m,
                                                                        cutout_size = 8)
"initialize dataset"
if args.data == 'cifar100':
    train_dataset = datasets.CIFAR100(f'{args.data_path}', train = True, transform = transform_tr, download = True)
    val_dataset = datasets.CIFAR100(f'{args.data_path}', train = False, transform = transform_val, download = True)
if args.data == 'cifar10':
    train_dataset = datasets.CIFAR10(f'{args.data_path}', train = True, transform = transform_tr, download = True)
    val_dataset = datasets.CIFAR10(f'{args.data_path}', train = False, transform = transform_val, download = True)
num_classes = len(train_dataset.classes)

"to train CIFAR100 using only 20 macroclasses it is necessary to modify the labels of the training an test set"
if args.superclasses:
    assert args.data=='cifar100'
    sup_tg = ut.c100super(train_dataset) #superclasses targets training set
    train_dataset = datasets.CIFAR100(f'{args.data_path}', train = True, transform = transform_tr, download = True,target_transform=lambda x: np.where(sup_tg==x)[0][0] )
    sup_tg = ut.c100super(val_dataset) #superclasses targets validation set
    val_dataset = datasets.CIFAR100(f'{args.data_path}', train = False, transform = transform_val, download = True, target_transform=lambda x: np.where(sup_tg==x)[0][0])
    num_classes = 20

"initialize dataloder"
if args.search:
    "used for a preliminary serach of the optimal set of hyperparams"
    train_subset, val_subset = ut.get_search_subsets(train_dataset, args.nimg_cl_tr, args.nimg_cl_val, nclasses = num_classes)
    args.epochs=100
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,num_workers=1, pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)



#-------------------------------------------------------------------------------
"initialize model"
net_dict = {'wide_resnet16': wide_resnet16,
            'wide_resnet10': wide_resnet10,
            'wide_resnet28': wide_resnet28,
            'wide_resnet40': wide_resnet40,}

model = net_dict[args.net_name](num_classes=num_classes, widening_factor = args.widening_factor, inplanes = args.inplanes)
device = 'cuda'
model = model.to(device)
if args.inplanes!=16: model_name = f'{args.net_name}_{args.widening_factor}_inpl{args.inplanes}'
else: model_name = f'{args.net_name}_{args.widening_factor}'



#-------------------------------------------------------------------------------
"initlialize optimizer and scheduler"
if args.label_smoothing:
    criterion = ut.LabelSmoothingCrossEntropy(smoothing =args.ls_magnitude)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 10**-5)



#-------------------------------------------------------------------------------
"optionally resume from a checkpoint"
path = f'{args.resume_folder}/{args.resume_filename}_model_cpt.pth.tar'
if args.resume:
    model, optimizer, scheduler, best_acc1, args.start_epoch = ut.resume(path, model, optimizer, scheduler)



#*******************************************************************************
"set the filenames according to the chosen training features"
print(f'model = {model_name};\n nclasses = {num_classes};\n lr = {args.lr};\n wd ={args.wd};\n batch size = {args.batch_size};\n opt = {optimizer.__class__.__name__};')
print(f'scheduler = {scheduler.__class__.__name__};')

filename = f'{args.data}_{model_name}_ep{args.epochs}_opt{optimizer.__class__.__name__}_sched{scheduler.__class__.__name__}_seed{args.seed}'
if args.pdrop is not None:
   filename = filename+f'_dropout{pdrop}'
   print(f'pdrop = {args.pdrop}')
if args.randaugment:
   filename = filename+f'_RandAugment{args.n}_{args.m}'
   print('randaugment')
if args.label_smoothing:
   filename = filename+f'_label_smoothing'
   print(f'label_smoothing {args.ls_magnitude}')
if args.superclasses:
   filename = filename+f'_super'
   print('cifar100 with superclasses')
if args.search:
   filename = filename+f'_search_lr{args.lr}_wd{args.wd}_ls{args.ls_magnitude}'
   print(f'searching for optimal hyperparams, training_size = {len(train_subset)//1000}k ')
if args.filename is not None:
    filename = filename+'_'+args.filename

"if the training resumes from previous checkpoint set the filename and path accordingly"
if args.resume:
    filename = args.resume_filename
    args.results_path = args.resume_folder

"files to same params and training stats"
if not args.resume:
    ut.write_params(model_name, optimizer, transform_tr.transforms, args.epochs, args.batch_size,
               scheduler = scheduler, folder = args.results_path, filename = f'/{filename}_params')
    ut.write_stats(folder = args.results_path, filename = f'/{filename}_stats', start= True)



#-------------------------------------------------------------------------------
"train the model"
tot_time = 0
best_acc1 =0
for epoch in range(args.start_epoch, args.epochs):

   print(f'epoch {epoch+1}/{args.epochs}')

   start = time.time()
   acc1_tr, loss_tr = train(train_loader, model, criterion, optimizer, epoch, scheduler)
   current_time = time.time()-start
   print(f'tr_acc = {acc1_tr: .5f},  tr_loss = {loss_tr: .5f}; time = {current_time} ')

   acc1_val, loss_val = validate(val_loader, model, criterion)
   tot_time += time.time()-start
   print(f'val_acc = {acc1_val: .5f}, val_loss =  {loss_val: .5f}; total_time = {tot_time}')

   if scheduler.__class__.__name__!='CosineAnnealingWarmRestarts' or 'CosineAnnealingLR':
       scheduler.step()

   is_best = acc1_val > best_acc1
   best_acc1 = max(acc1_val, best_acc1)
   ut.write_stats(epoch+1, acc1_val, acc1_tr, loss_val, loss_tr, folder = args.results_path, filename = f'/{filename}_stats')
   ut.write_results(epoch+1, best_acc1, tot_time, folder = args.results_path, filename = f'/{filename}_results')

   state = {'epoch': epoch + 1, 'arch': model_name, 'state_dict': model.state_dict(),
       'best_acc1': best_acc1, 'optimizer' : optimizer.state_dict(), 'scheduler': scheduler.state_dict() }

   ut.save_checkpoint(state, is_best, folder = args.results_path, filename = f'/{filename}_model')
