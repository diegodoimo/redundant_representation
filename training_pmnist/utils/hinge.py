def square_hinge(outputs, targets, margin = 1):
    batch_size = len(targets)
    delta = 1 - outputs*targets.view(batch_size, 1)
    delta[delta<0]=0
    delta = delta**2
    loss = 0.5*torch.sum(delta)/batch_size
    return loss

def train_hinge_loss(images, target, model, criterion, optimizer, batch_size):

    # switch to train mode
    model.train()
    output = model(images)
    loss = criterion(output, target)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()

    # measure accuracy and record loss
    pre_accuracy = output.view(-1)*target.view(-1)
    accuracy = len(pre_accuracy[pre_accuracy>0])/batch_size

    return accuracy, loss.item()

def val_hinge_loss(images, target, model, criterion, batch_size):

    # switch to train mode
    model.eval()
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    pre_accuracy = output.view(-1)*target.view(-1)
    accuracy = len(pre_accuracy[pre_accuracy>0])/batch_size

    return accuracy, loss.item()

def train_hinge_loss_batch(images, target, model, criterion, optimizer, batch_size):

    # switch to train mode
    model.train()
    nbatches = len(images)//batch_size
    accuracy = torch.empty(nbatches)
    losses = torch.empty(nbatches)

    for i in range(nbatches):

        output = model(images[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, target[i*batch_size:(i+1)*batch_size])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        losses[i] = loss.item()

        # measure accuracy and record loss
        pre_accuracy = output.view(batch_size)*target[i*batch_size:(i+1)*batch_size].view(batch_size)
        accuracy[i] = len(pre_accuracy[pre_accuracy>0])/batch_size

    if nbatches*batch_size!=len(images):
        output = model(images[nbatches*batch_size:])
        loss = criterion(output, target[nbatches*batch_size:])

        optimizer.zero_grad()
        loss.backward()
        losses[nbatches] = loss.item()

        pre_accuracy = output.view(batch_size)*target[i*batch_size:(i+1)*batch_size]
        accuracy[nbatches] = len(pre_accuracy[pre_accuracy>0])/batch_size

    return torch.mean(accuracy).item(), torch.mean(losses).item()

def val_hinge_loss_batch(images, target, model, batch_size):

    # switch to train mode
    model.eval()
    nbatches = len(images)//batch_size
    accuracy = torch.empty(nbatches)
    losses = torch.empty(nbatches)

    for i in range(nbatches):

        output = model(images[i*batch_size:(i+1)*batch_size])
        loss = criterion(output, target[i*batch_size:(i+1)*batch_size])
        losses[i] = loss.item()

        # measure accuracy and record loss
        pre_accuracy = output.view(batch_size)*target[i*batch_size:(i+1)*batch_size].view(batch_size)
        accuracy[i] = len(pre_accuracy[pre_accuracy>0])/batch_size

    if nbatches*batch_size!=len(images):
        output = model(images[nbatches*batch_size:])
        loss = criterion(output, target[nbatches*batch_size:])
        losses[nbatches] = loss.item()

        pre_accuracy = output.view(batch_size)*target[i*batch_size:(i+1)*batch_size]
        accuracy[nbatches] = len(pre_accuracy[pre_accuracy>0])/batch_size

    return torch.mean(accuracy).item(), torch.mean(losses).item()


def train_cross_entropy(images, targets, model, criterion, optimizer, batch_size):

    # switch to train mode
    model.train()
    out = model(images)
    loss = criterion(out, targets.view(-1))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()

    # measure accuracy and record loss
    _, pred = out.topk(1)                               #(batch_size x 1)  max indici logit
    correct = torch.eq(targets.view(-1), pred.view(-1))          #pred--> (batch_size)
    accuracy = torch.sum(correct).item()/len(images)    #ultimo batch può contenere meno immagini...

    return accuracy, loss.item(), out

#*******************************************************************************
def val_cross_entropy(images, targets, model, batch_size):

    # switch to train mode
    model.train()
    out = model(images)
    loss = criterion(out, targets.view(-1))

    # measure accuracy and record loss
    _, pred = out.topk(1) #(batch_size x 1)  max indici logit
    correct = torch.eq(targets.view(-1), pred.view(-1)) #pred--> (batch_size)
    accuracy = torch.sum(correct).item()/len(images) #ultimo batch può contenere meno immagini...

    return accuracy, loss.item()


def train_cross_entropy_batch(dataloader, model, criterion, optimizer, batch_size):
    model.train()

    accuracy = torch.empty(len(dataloader))
    losses = torch.empty(len(dataloader))

    images = next(iter(dataloader))[0].cuda(None, non_blocking=True)
    targets = next(iter(dataloader))[1].cuda(None, non_blocking=True)

    for i, (images_, targets_) in enumerate(dataloader):
        images_ = images_.cuda(None, non_blocking=True)           #batch_size x image_shape
        targets_ = targets_.cuda(None, non_blocking=True)         #(batch_size)

        #print(torch.all(images == images_))
        #print(torch.all(targets == targets_))

        out = model(images) #batch_size x nclasses

        loss = criterion(out, targets)
        losses[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()

        #accuracy
        _, pred = out.topk(1)                                   #(batch_size x 1)  max indici logit
        correct = torch.eq(targets, pred.view(-1))              #pred--> (batch_size)
        accuracy[i] = torch.sum(correct).item()/len(images)     #ultimo batch può contenere meno immagini...

    return torch.mean(accuracy).item(), torch.mean(loss).item(), out

def val_cross_entropy_batch(dataloader, model, batch_size):
    model.eval()

    accuracy = torch.empty(len(dataloader))
    losses = torch.empty(len(dataloader))

    for i, (images, targets) in enumerate(dataloader):
        images = images.cuda(None, non_blocking=True)  #batch_size x image_shape
        targets = targets.cuda(None, non_blocking=True) #(batch_size)
        out = model(images) #batch_size x nclasses

        loss = criterion(out, targets)
        losses[i] = loss.item()

        #accuracy
        _, pred = out.topk(1) #(batch_size x 1)  max indici logit
        correct = torch.eq(targets, pred.view(-1)) #pred--> (batch_size)
        accuracy[i] = torch.sum(correct).item()/len(images) #ultimo batch può contenere meno immagini...

    return torch.mean(accuracy).item(), torch.mean(loss).item()
