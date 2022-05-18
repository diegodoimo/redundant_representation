import torch
import numpy as np
import os

#--------------------------
import layer_loader
import my_utils as ut
import hinge

#*******************************************************************************
def get_activations(images, model, device, module):

    if module == 'input': hout = images
    else:
        hout = []
        def hook(module, input, output):
            hout.append(output)
        handle = module.register_forward_hook(hook)
        output = model(images)
        del output
        hout = hout[0]
        handle.remove()

    act = hout.view(images.shape[0], -1).detach().cpu()

    hout = hout.detach().cpu()
    del hout

    return act


#*******************************************************************************
def get_features(model, images):
    modules, names, depths = layer_loader.get_fully_connected_biroli_depths(model)
    for i, module_ in enumerate(modules):
        if i==len(modules)-2:
            module = module_
            print(f'estractiong features layer {names[i]}')

    hout = []
    def hook(module, input, output):
        hout.append(output)
    handle = module.register_forward_hook(hook)
    output = model(images)

    hout = hout[0]
    handle.remove()
    features = hout.view(images.shape[0], -1)

    hout = hout.detach().cpu()
    del hout
    del output

    model.train()
    return features

def compute_chunk_acc(model, images,targets, widths):
    model.eval()
    weights = model.state_dict()['linear_out.weight']
    bias = model.state_dict()['linear_out.bias']
    features = get_features(model, images)
    W = features.shape[1]

    acc = []
    for w in widths:
        if w < W:
            acc_tmp = []
            for i in range(100):
                ind = np.random.choice(W, size = w, replace = False)
                outputs = torch.matmul(features[:, ind], weights[:, ind].T)+bias
                accuracy = 1.*torch.sum(outputs.reshape(-1)*targets>0)
                acc_tmp.append(accuracy.detach().cpu().numpy())
            acc.append(np.mean(acc_tmp)/len(targets))
    return acc


def train_hinge_loss_batch(images, target, model, criterion, optimizer, batch_size, cross_entropy=False, compute_chunk=False, widths = None):
    if compute_chunk: assert widths is not None


    # switch to train mode
    model.train()
    shuffled_ind = torch.randperm(images.shape[0])
    images_ = images[shuffled_ind]
    target_ = target[shuffled_ind]

    data_batches = torch.split(images_, batch_size)
    target_batches = torch.split(target_, batch_size)
    #print(data_batches.shape)
    accuracy = 0
    losses = 0

    for i in range(len(data_batches)):
        #print(data_batches[i].shape)
        output = model(data_batches[i])
        loss = criterion(output, target_batches[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses+= loss.item()*data_batches[i].shape[0]

        pre_accuracy = output.view(-1)*target_batches[i].view(-1)
        accuracy+= len(pre_accuracy[pre_accuracy>0])

    if compute_chunk: chunk_acc = compute_chunk_acc(model, images, target, widths)
    else: chunk_acc = None

    return accuracy/len(images), losses/len(images), chunk_acc

def train_hinge_loss(images, target, model, criterion, optimizer, batch_size, cross_entropy, compute_chunk=False, widths = None):

    if compute_chunk: assert widths is not None

    model.train()
    output = model(images)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if cross_entropy:
        _, pred = output.topk(1)
        correct = torch.eq(target.view(-1), pred.view(-1))
        accuracy = torch.sum(correct).item()/len(images)
    else:
        pre_accuracy = output.view(-1)*target.view(-1)
        accuracy = len(pre_accuracy[pre_accuracy>0])/batch_size

    if compute_chunk: chunk_acc = compute_chunk_acc(model, images, target, widths)
    else: chunk_acc = None

    return accuracy, loss.item(), chunk_acc

def val_hinge_loss(images, target, model, criterion, batch_size, cross_entropy, compute_chunk=False, widths = None):

    model.eval()
    output = model(images)
    loss = criterion(output, target)

    if cross_entropy:
        _, pred = output.topk(1)                               #(batch_size x 1)  max indici logit
        correct = torch.eq(target.view(-1), pred.view(-1))          #pred--> (batch_size)
        accuracy = torch.sum(correct).item()/len(images)    #ultimo batch può contenere meno immagini...
    else:
        pre_accuracy = output.view(-1)*target.view(-1)
        accuracy = len(pre_accuracy[pre_accuracy>0])/batch_size

    if compute_chunk: chunk_acc = compute_chunk_acc(model, images, target, widths)
    else: chunk_acc = None

    return accuracy, loss.item(), chunk_acc





def square_hinge(outputs, targets, margin = 1):
    batch_size = len(targets)
    delta = 1 - outputs*targets.view(batch_size, 1)
    delta[delta<0]=0
    delta = delta**2
    loss = 0.5*torch.sum(delta)/batch_size
    return loss

def register_state(images, target, model, criterion, batch_size_val, optimizer, cross_entropy, tr_acc,
                            tr_loss, epoch, best_acc, chunk_acc_tr, args, compute_chunk_acc = False, widths = None,):

    val_acc, val_loss, c_acc_ts = val_hinge_loss(images, target, model, criterion, batch_size_val,
                        cross_entropy, compute_chunk = compute_chunk_acc, widths = widths)
    #val_acc, val_loss = val_hinge_loss_batch(images, target, model, criterion, batch_size_val, nbatches, nbatches_full, reminder, w, wlast, cross_entropy)
    #val_acc, val_loss = val_hinge_loss_batch(images, target, model, criterion, batch_size_val)
    #gradient = ut.get_gradient_norm(optimizer)

    state = {   'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'tr_acc': tr_acc,
                'tr_loss': tr_loss,
                'val_acc': val_acc,
                'val_loss': val_loss}


    is_best = val_acc > best_acc
    if is_best: best_acc = max(val_acc, best_acc)


    ut.save_checkpoint(state, is_best, folder = args.results,  filename = f'/{args.filename}')
    write_stats(epoch, val_acc, tr_acc, val_loss, tr_loss, chunk_acc_tr = chunk_acc_tr, chunk_acc_ts = c_acc_ts,
                                folder = args.results, filename = f'/{args.filename}_stats')

    ut.write_results(epoch, best_acc, folder = args.results, filename = f'/{args.filename}_results')

    return val_acc, val_loss, best_acc, c_acc_ts


def write_stats(epoch = None, val_acc = None, tr_acc = None, val_loss = None, tr_loss = None,
                chunk_acc_tr = None, w = None, chunk_acc_ts =None, folder = None ,filename = None, start=False):
    if not os.path.isdir(f'{folder}'): pathlib.Path(f'{folder}').mkdir(parents=True, exist_ok=True)

    if start:
        action = 'w'
        with open(f'{folder}/{filename}', action) as f:
            f.write('epoch test_acc train_acc test_loss train_loss')
            if w is not None:
                for i in range(len(w)):
                    f.write(f' c_width: {w[i]}')
            f.write('\n')

    else:
        action = 'a'
        with open(folder+filename, action) as f:
            f.write(f'{epoch} {val_acc: .7f} {tr_acc: .7f} {val_loss: .7f} {tr_loss: .7f}')
            if chunk_acc_tr is not None:
                for i in range(len(chunk_acc_tr)):
                    f.write(f' {chunk_acc_tr[i]: .7f}')
            if chunk_acc_ts is not None:
                for i in range(len(chunk_acc_ts)):
                    f.write(f' {chunk_acc_ts[i]: .7f}')

            f.write('\n')

def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']+1


def load_params(model, model_child, width, bottleneck, layer_bottleneck):

    state_dict = model.state_dict()
    key_list = list(model.state_dict().keys())

    for i, key in enumerate(state_dict):
        #each layer has weight and biases
        if i//2==layer_bottleneck-1:
            random_act = torch.randperm(int(width))[:bottleneck]
            #weight in
            state_dict[key] = state_dict[key][random_act]
            #bias in
            state_dict[key_list[i+1]] = state_dict[key_list[i+1]][random_act]
            #weight out
            state_dict[key_list[i+2]] = state_dict[key_list[i+2]][:, random_act]
            break

    model_child.load_state_dict(state_dict)

    return model_child

#*******************************************************************************
def build_batches(nimages, batch_size):

    nbatches_full = nimages//batch_size
    reminder = nimages%batch_size
    if reminder!=0:
        nbatches = nbatches_full+1
        w = nbatches_full*batch_size/nimages
        wlast = reminder/nimages
    else:
        nbatches = nbatches_full
        w = batch_size*(nbatches-1)/nimages
        wlast = batch_size/nimages

    return nbatches, nbatches_full, reminder, w, wlast

#to be checked
def val_hinge_loss_batch(images, targets, model, criterion, batch_size):
    print('to be checked')
    # switch to train mode
    ntot = len(images)

    nlast_batch = ntot%batch_size
    if nlast_batch==0:
        nbatches = ntot//batch_size
        nlast_batch = batch_size
    else: nbatches = ntot//batch_size +1

    perm = torch.randperm(images.shape[0])
    accuracy_avg = 0
    losses_avg = 0

    model.eval()

    for i in range(nbatches):

        if i == nbatches-1:
            indices = perm[i*batch_size:]
            w = 1.0*nlast_batch/ntot
        else:
            indices = perm[i*batch_size:(i+1)*batch_size]
            w = 1.0*batch_size/ntot

        output = model(images[indices])
        loss = criterion(output, targets[indices])

        _, predicted = output.max(1)
        accuracy_avg += w*predicted.eq(targets[indices]).sum().item()
        losses_avg+= w*loss.item()

    return accuracy_avg, losses_avg


def val_hinge_loss_batch(images, target, model, criterion, batch_size, nbatches, nbatches_full, reminder, w, wlast, cross_entropy):

    # switch to train mode
    model.eval()
    accuracy = torch.empty(nbatches)
    losses = torch.empty(nbatches)

    for i in range(nbatches_full):

        output = model(images[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, target[i*batch_size:(i+1)*batch_size])
        losses[i] = loss.item()

        if cross_entropy:
            _, pred = output.topk(1)
            correct = torch.eq(target[i*batch_size:(i+1)*batch_size].view(-1), pred.view(-1))
            accuracy[i] = torch.sum(correct).item()/batch_size
        else:
            pre_accuracy = output.view(batch_size)*target[i*batch_size:(i+1)*batch_size].view(batch_size)
            accuracy[i] = len(pre_accuracy[pre_accuracy>0])/batch_size

    if reminder!=0:
        output = model(images[nbatches*batch_size:])
        loss = criterion(output, target[nbatches*batch_size:])
        losses[nbatches] = loss.item()

        if cross_entropy:
            _, pred = output.topk(1)
            correct = torch.eq(target[nbatches_full*batch_size:].view(-1), pred.view(-1))
            accuracy[nbatches-1] = torch.sum(correct).item()/reminder
        else:
            pre_accuracy = output.view(reminder)*target[nbatches_full*batch_size:].view(reminder)
            accuracy[nbatches-1] = len(pre_accuracy[pre_accuracy>0])/reminder

    if len(accuracy)==1:
        accuracy_avg = torch.mean(accuracy[-1]).item()*wlast
        losses_avg = torch.mean(losses[-1]).item()*wlast
    else:
        accuracy_avg = torch.mean(accuracy[:-1]).item()*w + torch.mean(accuracy[-1]).item()*wlast
        losses_avg = torch.mean(losses[:-1]).item()*w + torch.mean(losses[-1]).item()*wlast

    return accuracy_avg, losses_avg








# def load_params(model, model_child, width, bottleneck):
#
#     #first_layer
#     model_child.linear_in.load_state_dict(model.linear_in.state_dict())
#
#     #central block
#     body_pretrained = model.linear_body.state_dict()
#     body_child_dict = model_child.linear_body.state_dict()
#     pretrained_dict = {k: v for k, v in body_pretrained.items() if k in body_child_dict}
#     body_child_dict.update(pretrained_dict)
#     model_child.linear_body.load_state_dict(body_child_dict)
#
#     random_act = torch.randperm(int(width))[:bottleneck]
#     lastw = body_pretrained[list(body_pretrained.keys())[-2]]
#     lastb = body_pretrained[list(body_pretrained.keys())[-1]]
#     dict_bottleneck = {'0.weight': lastw[random_act], '0.bias': lastb[random_act]}
#     model_child.bottleneck.load_state_dict(dict_bottleneck)
#
#     #output_layer
#     dict_out = {
#         'weight': model.linear_out.state_dict()['weight'][:, random_act],
#         'bias': model.linear_out.state_dict()['bias']
#         }
#     model_child.linear_out.load_state_dict(dict_out)
#
#     return model_child
