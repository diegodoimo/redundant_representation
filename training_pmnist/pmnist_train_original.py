import torch
import torchvision.transforms as transforms
from torch.optim import Adam
import torchvision.datasets as datasets
import argparse
from my_nets.biroli_scaling import fully_connected_new, fully_connected
from NNmodules import my_dataset, layer_loader
import numpy as np
import training_utils.my_utils as ut
import training_utils.hinge as hinge
import time
from training_utils.my_utils import reduce
import copy
import utils as ut_
import math
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
#*******************************************************************************
def set_seed(seed):
    torch.backends.cudnn.deterministic = True       #computation can use nondeterministic algorithms
    torch.backends.cudnn.benchmark = False          #fastest convolutional algorithm is selected if true (better choice but is non deterministc!)
    torch.manual_seed(seed)                         #seed for both cpu and cuda
def tg_targets(tg):
    if tg==-1: tg=0
    return tg

#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR', default = '/home/diego/ricerca/datasets/mnist/MNIST_pc/std_not_rescaled/10pc',
                    help='path to dataset')

parser.add_argument('--width', default=512, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nlayers', default=6, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--pdrop', default=None, type=float)
parser.add_argument('--wd', default=0.03, type=float)
parser.add_argument('--lr', default=0.03, type=float)

parser.add_argument('--epochs', default=2*10**4, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--trainset', action='store_true')

parser.add_argument('-b', '--batch_size', default=1024, type=int,)

parser.add_argument('--seed', default=1602, type=int, help='seed for initializing training. ')
parser.add_argument('--cpt', metavar = 'DIR', default = './results/data/models/20kepochs',help = 'path where to save results')
parser.add_argument('--results', metavar = 'DIR', default = './results',help = 'path where to save results')
parser.add_argument('--filename', default = '/trial')
args = parser.parse_args()

#*******************************************************************************
nlayers = args.nlayers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
width = args.width

lr = args.lr
bs = args.bs
with_batch_norm = args.batch_norm
weight_decay = args.wd
trainset = args.trainset
cross_entropy = False
input_size = 10
#*******************************************************************************
set_seed(args.seed)
if with_batch_norm:args.filename+='_batch_norm'
if bs is not None: args.filename+=f'_bs{bs}'
# if dropout:args.filename+='_dropout'
if trainset: args.filename+='_60ktrain'

if cross_entropy ==True:
    output_size = 2
    criterion = torch.nn.CrossEntropyLoss().cuda() #output units must be two
    trainset = my_dataset.MNIST_pc(f'{args.data}', train=False, target_transform = tg_targets)
    valset = my_dataset.MNIST_pc(f'{args.data}', train=True, target_transform = tg_targets)
else:
    output_size = 1
    criterion = ut_.square_hinge
    trainset = my_dataset.MNIST_pc(f'{args.data}', train=trainset)
    valset = my_dataset.MNIST_pc(f'{args.data}', train=trainset)
#*******************************************************************************
nparams = round(width)**2*(nlayers-2)+(input_size+output_size)*round(width)

#train full model

model = fully_connected_new(round(width), depth = nlayers, input_size = input_size, output_size = output_size,
                                    dropout = False, batch_norm = with_batch_norm, orthog_init = True)
model.to(device)
model_name = model.__class__.__name__+f'_{width}_nlay{nlayers}'

#*******************************************************************************
#for other training options see parity mnist final
#*******************************************************************************
# min_lr, max_lr = math.log(10**-3.2), math.log(10**-2)
# lr = lr = math.exp(torch.distributions.uniform.Uniform(min_lr, max_lr).sample().item())
#
# min_wd, max_wd = math.log(10**-2), math.log(10**-1)
# weight_decay = math.exp(torch.distributions.uniform.Uniform(min_wd, max_wd).sample().item())
#
#momentum = 0.9
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov = False)
#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5000, T_mult=1, eta_min =0)
#args.filename += f'_sgd'


lr = min(10**-4, (nparams/(nlayers-2))**-0.75/10)
#lr = min(10**-4, (width**-1.5)/10)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = None
criterion = ut_.square_hinge
best_acc = 0

#*******************************************************************************
batch_size_tr = len(trainset)
batch_size_val = len(valset)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, pin_memory=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=False,  pin_memory=True, num_workers=0)

#they apply deterministic learning scheme
train_images = next(iter(train_loader))[0].cuda(None, non_blocking=True)
train_targets = next(iter(train_loader))[1].cuda(None, non_blocking=True).float()
val_images = next(iter(val_loader))[0].cuda(None, non_blocking=True)
val_targets = next(iter(val_loader))[1].cuda(None, non_blocking=True).float()

#*******************************************************************************
width_chunks = [2**i for i in range(10)]
ut.write_params(model = model_name, optimizer = optimizer, epochs = args.epochs, batch_size = batch_size_tr, scheduler = scheduler,
                      folder = args.results, filename = f'/{args.filename}_params')
if not args.resume:
    ut_.write_stats(folder = args.results, filename = f'/{args.filename}_stats', start= True, w = width_chunks)

print(f'seed = {args.seed}')
print(f'width = {width}')
print(f' weight_decay = {weight_decay}')
print(f'batch_norm = {with_batch_norm}')
# #*****************************************************************************
ep_cpt = 500
for epoch in range(args.start_epoch, args.epochs):

    if (epoch+1)%ep_cpt==0: compute_chunk_acc = False
    else: compute_chunk_acc= False

    if bs is None:
        tr_acc, tr_loss, c_acc_tr = ut_.train_hinge_loss(train_images, train_targets, model, criterion,
                                                     optimizer, batch_size_tr, cross_entropy, compute_chunk_acc, width_chunks)
    else:
        tr_acc, tr_loss, c_acc_tr = ut_.train_hinge_loss_batch(train_images, train_targets, model, criterion,
                                                     optimizer, bs, cross_entropy, compute_chunk_acc, width_chunks)

    if scheduler is not None: scheduler.step(epoch)

    if (epoch+1)%ep_cpt==0:
        val_acc, val_loss, best_acc, c_acc_ts = ut_.register_state(val_images, val_targets, model,
            criterion, batch_size_val, optimizer, cross_entropy, tr_acc, tr_loss, epoch+1, best_acc,
            chunk_acc_tr = c_acc_tr, compute_chunk_acc = compute_chunk_acc, widths=width_chunks, args = args)

        print(f'epoch {(epoch+1)/1000}k/{args.epochs/1000}k')
        print(f'tr_acc = {np.mean(tr_acc): .5f}  tr_loss = {np.mean(tr_loss): .7f}')
        print(f'val_acc = {np.mean(val_acc): .5f}  val_loss =  {np.mean(val_loss): .5f}')

    if compute_chunk_acc:
        print('chunk_accuracy')
        for i in range(len(c_acc_tr)):  print(f'{width_chunks[i]}: {c_acc_tr[i]} {c_acc_ts[i]}')

    if tr_loss < torch.finfo().eps*10:
        val_acc, val_loss, best_acc, c_acc_ts = ut_.register_state(val_images, val_targets, model,
            criterion, batch_size_val, optimizer, cross_entropy, tr_acc, tr_loss, epoch+1, best_acc,
            chunk_acc_tr = c_acc_tr, compute_chunk_acc = compute_chunk_acc, widths=width_chunks, args = args)
        print(f'seed = {args.seed} finished in {epoch} epochs')
        break

    elif epoch+1 == args.epochs and epoch%ep_cpt!=0:
        val_acc, val_loss, best_acc, c_acc_ts = ut_.register_state(val_images, val_targets, model,
            criterion, batch_size_val, optimizer, cross_entropy, tr_acc, tr_loss, epoch+1, best_acc,
            chunk_acc_tr = c_acc_tr, compute_chunk_acc = compute_chunk_acc, widths=width_chunks, args = args)
