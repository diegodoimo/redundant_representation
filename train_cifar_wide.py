import argparse
import random
import time
import numpy as np
from contextlib import suppress

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.wide_resnet import wide_resnet16, wide_resnet28
from utils.densenet import  densenet40
import utils.my_utils as ut

#*******************************************************************************
def train(loader, model, criterion, optimizer, device, epoch = None, scheduler = None, amp_autocast=suppress, loss_scaler=None,):

    second_order = False

    model.train()
    tr_loss = 0
    correct = 0
    total = 0
    iters = len(loader)

    for i, (inputs, targets) in enumerate(loader):

        start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=loss_scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if loss_scaler is not None:
            loss_scaler.scale(loss).backward(create_graph=second_order)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tr_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return 1.0*correct/total, tr_loss/total

def validate(loader, model, criterion, device, amp_autocast=suppress):

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 1.0*correct/total, val_loss/total

#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--data', default = 'cifar10', metavar='DIR')
parser.add_argument('--data_path', metavar='DIR', default ='.')#'/home/diego/ricerca/datasets'

parser.add_argument('--net_name', default = 'wide_resnet16',)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--inplanes', default=16, type=int)
parser.add_argument('--widening_factor', default=1, type=int)
parser.add_argument('--last_layer_width', default = None, type=float)
parser.add_argument('--last_block_width', default = None, type=float)
parser.add_argument('--stem_type', default = 'cifar')
parser.add_argument('--kernel_size', default = 3, type = int)

parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr',  default=0.1, type=float)
parser.add_argument('--weight_decay',  default=5e-4, type=float, )

parser.add_argument('--ls_magnitude', default=0.0, type=float)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--use_amp', action='store_true')

parser.add_argument('--memory_usage', action='store_true')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--results_path', metavar = 'DIR', default = './models')
parser.add_argument('--resume_folder', metavar = 'DIR', default = None)
parser.add_argument('--resume_filename', metavar = 'DIR', default = None)
parser.add_argument('--filename')

args = parser.parse_args()
#*******************************************************************************
if args.seed is not None:
   random.seed(args.seed)
   torch.manual_seed(args.seed)
#*******************************************************************************
# Data loading code
if args.data == 'cifar100':
    args.data_path = args.data_path+'/cifar100'
    norm_mean = (0.5071, 0.4865, 0.4409)
    norm_std = (0.2673, 0.2564, 0.2762)
    num_classes = 100
elif args.data == 'cifar10':
    args.data_path = args.data_path+'/cifar10'
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.247, 0.243, 0.261)
    num_classes = 10

transform_tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode ='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
                ])

transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

if args.data == 'cifar100':
    train_dataset = datasets.CIFAR100(f'{args.data_path}', train = True, transform = transform_tr, download = True)
    val_dataset = datasets.CIFAR100(f'{args.data_path}', train = False, transform = transform_val, download = True)
if args.data == 'cifar10':
    train_dataset = datasets.CIFAR10(f'{args.data_path}', train = True, transform = transform_tr, download = True)
    val_dataset = datasets.CIFAR10(f'{args.data_path}', train = False, transform = transform_val, download = True)

num_classes = len(train_dataset.classes)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                pin_memory=True)

#-------------------------------------------------------------------------------
net_dict = {'wide_resnet16': wide_resnet16,
            'wide_resnet28': wide_resnet28,
            'densenet40': densenet40
            }

if args.net_name.startswith('densenet'):
    model = net_dict[args.net_name](num_init_features=args.inplanes,
                                growth_rate = args.inplanes//2,
                                stem_type = args.stem_type,
                                kernel_size = args.kernel_size)
    model_name = f'{args.net_name}_{args.inplanes}'

elif args.net_name.startswith('wide_resnet'):
    model = net_dict[args.net_name](num_classes=num_classes,
                        widening_factor = args.widening_factor,
                        inplanes = args.inplanes)

    if args.inplanes!=16:
        model_name = f'{args.net_name}_{args.widening_factor}_inpl{args.inplanes}'
    else:
        model_name = f'{args.net_name}_{args.widening_factor}'

device = 'cuda'
model = model.to(device)
#*******************************************************************************
criterion = ut.LabelSmoothingCrossEntropy(smoothing =args.ls_magnitude)

#-------------------------------------
#automatic mixed precision training
amp_autocast = suppress
loss_scaler = None
if args.use_amp:
    loss_scaler = torch.cuda.amp.GradScaler()
    amp_autocast = torch.cuda.amp.autocast

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 10**-5)
best_acc1 =0

if args.memory_usage:
    ut.print_memory_footprint(model, val_loader, model_name, args, device, criterion, optimizer)
#*******************************************************************************
if args.resume:
    path = f'{args.resume_folder}/{args.resume_filename}_model_cpt.pth.tar'
    model, optimizer, scheduler, best_acc1, args.start_epoch = ut.resume(path, model, optimizer, scheduler)

#*******************************************************************************
print(f'model = {model_name};\n nclasses = {num_classes};\n lr = {args.lr};\n wd ={args.weight_decay};\nbatch size = {args.batch_size};\n')

filename = f'{args.data}_{model_name}_ep{args.epochs}_seed{args.seed}_wd{args.weight_decay}_ls{args.ls_magnitude}'
if args.filename is not None:
    filename = filename+'_'+args.filename

if args.resume:
    filename = args.resume_filename
    args.results_path = args.resume_folder

if not args.resume:
    ut.write_stats(folder = args.results_path, filename = f'/{filename}_stats', start= True)
#-------------------------------------------------------------------------------
tot_time = 0
for epoch in range(args.start_epoch, args.epochs):

    print(f'epoch {epoch+1}/{args.epochs}')

    start = time.time()
    acc1_tr, loss_tr = train(train_loader, model, criterion, optimizer, device, epoch, scheduler, amp_autocast, loss_scaler)
    current_time = time.time()-start
    print(f'tr_acc = {acc1_tr: .5f},  tr_loss = {loss_tr: .5f}; time = {current_time} ')

    acc1_val, loss_val = validate(val_loader, model, criterion, device, amp_autocast)
    tot_time += time.time()-start
    print(f'val_acc = {acc1_val: .5f}, val_loss =  {loss_val: .5f}; total_time = {tot_time}')

    scheduler.step()

    is_best = acc1_val > best_acc1
    best_acc1 = max(acc1_val, best_acc1)
    ut.write_stats(epoch+1, acc1_val, acc1_tr, loss_val, loss_tr, folder = args.results_path, filename = f'/{filename}_stats')
    ut.write_results(epoch+1, best_acc1, tot_time, folder = args.results_path, filename = f'/{filename}_results')


    state = {'epoch': epoch + 1, 'arch': model_name, 'state_dict': model.state_dict(),
       'best_acc1': best_acc1, 'optimizer' : optimizer.state_dict(), 'scheduler': scheduler.state_dict() }

    if args.save_checkpoint and epoch+1 in [1, 2, 5, 10, 20, 40, 60, 80, 100, 130, 140, 150, 160, 180]:
        ut.save_checkpoint(state, is_best, folder = args.results_path, filename = f'/{filename}_model', register_epoch = epoch+1)
    ut.save_checkpoint(state, is_best, folder = args.results_path, filename = f'/{filename}_model')
