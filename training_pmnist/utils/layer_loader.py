def update_conv_count(block, count, downsample=True):
    for c in block.children():
        name = c.__class__.__name__
        if 'Conv' in name:
            count += 1
        if name=='Sequential' and downsample:
            count = update_conv_count(c, count, downsample)
    return count

def getLayerDepth(group):
    count = 0
    for block in group:
        count = update_conv_count(block, count)
    return count

def add_module(name, module, depth, names, modules, depths):
    names.append(name)
    modules.append(module)
    depths.append(depth)
    return names, modules, depths

#*******************************************************************************
#for resnets groups
def add_submodules(block_name, block, downsample, count, red, names, modules, depths):
    nblocks = len(list(block))
    for i, module in enumerate(block):
        count += update_conv_count(module, 0, downsample)
        if (i+1) == nblocks:
            names, modules, depths = add_module(block_name, module, count, names, modules, depths)
            continue
        if (i+1)%red==0:
            names, modules, depths = add_module(module.__class__.__name__, module, count, names, modules, depths)

    return names, modules, depths, count

def getWideResNetDepths(model, red, downsample = True):

    modules = []
    names = []
    depths = []

    count = 0
    names, modules, depths = add_module('input', 'input', count, names, modules, depths)
    count += 1
    names, modules, depths = add_module('conv1', model.conv0, count, names, modules, depths)

    names, modules, depths, count = add_submodules('group0', model.group0, downsample, count, red, names, modules, depths)
    names, modules, depths, count = add_submodules('group1', model.group1, downsample, count, red, names, modules, depths)
    names, modules, depths, count = add_submodules('group2', model.group2, downsample, count, red, names, modules, depths)

    names, modules, depths = add_module('bn', model.bn, count, names, modules, depths)
    names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
    count += 1
    names, modules, depths = add_module('fc', model.fc, count, names, modules, depths)

    names = [x for _, x in sorted(zip(depths,names), key=lambda x: x[0])]
    modules = [x for _, x in sorted(zip(depths,modules), key=lambda x: x[0])]
    depths = [x for _, x in sorted(zip(depths,depths), key=lambda x: x[0])]

    return modules, names, depths


# from my_nets.wide_resnet import wide_resnet16
# net = wide_resnet16()
# modules, names, depths = getWideResNetDepths(net, red=1, downsample = True)

#*******************************************************************************
def getResNetsDepths(model, red, downsample = False):

    modules = []
    names = []
    depths = []

    count = 0
    names, modules, depths = add_module('input', 'input', count, names, modules, depths)
    count+=1
    names, modules, depths = add_module('maxpool', model.maxpool, count, names, modules, depths)

    names, modules, depths, count = add_submodules('group0', model.layer1, downsample, count, red, names, modules, depths)
    names, modules, depths, count = add_submodules('group1', model.layer2, downsample, count, red, names, modules, depths)
    names, modules, depths, count = add_submodules('group2', model.layer3, downsample, count, red, names, modules, depths)

    if hasattr(model, "layer4"):  #in an example i removed the group3
        names, modules, depths, count = add_submodules('group3', model.layer4, downsample, count, red, names, modules, depths)

    names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
    count += 1
    names, modules, depths = add_module('fc', model.fc, count, names, modules, depths)

    names = [x for _,x in sorted(zip(depths,names), key=lambda x: x[0])]
    modules = [x for _,x in sorted(zip(depths,modules), key=lambda x: x[0])]
    depths = [x for _,x in sorted(zip(depths,depths), key=lambda x: x[0])]

    return modules, names, depths

# from torchvision.models import resnet18
# net = resnet18()
# modules, names, depths = getResNetsDepths(net, red = 1)

#*******************************************************************************
def getvggDepths(model, red=2):
    count = 0
    modules = []
    names = []
    depths = []
    names, modules, depths = add_module('input', 'input', 0, names, modules, depths)
    ngroup = 0
    for i,module in enumerate(model.features):
        name = module.__class__.__name__
        if 'Conv2d' in name:
            count += 1
            if red==1: names, modules, depths = add_module(name, module, count, names, modules, depths)
        if 'MaxPool2d' in name:
            ngroup+=1
            if ngroup==5 and red>1:continue
            else: names, modules, depths = add_module(name, module, count, names, modules, depths)
    names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)

    for i,module in enumerate(model.classifier):
        name = module.__class__.__name__
        if 'Linear' in name:
            count += 1
            names, modules, depths = add_module(name, module, count, names, modules, depths)

    return modules, names, depths

#*******************************************************************************
def getgooglenetdepths(model):
    count = 0
    modules = []
    names = []
    depths = []

    #input
    names, modules, depths = add_module('input', 'input', 0, names, modules, depths)

    #MaxPool
    count+=1
    names, modules, depths = add_module('maxpool1', model.maxpool1, count, names, modules, depths)

    count+=2
    names, modules, depths = add_module('maxpool2', model.maxpool2, count, names, modules, depths)

    #Inception blocks
    count+=2
    names, modules, depths = add_module('inception3a', model.inception3a, count, names, modules, depths)
    count+=2
    #names, modules, depths = add_module('inception3b', model.inception3b, count, names, modules, depths)
    names, modules, depths = add_module('maxpool3', model.maxpool3, count, names, modules, depths)

    count+=2
    names, modules, depths = add_module('inception4a', model.inception4a, count, names, modules, depths)
    count+=2
    names, modules, depths = add_module('inception4b', model.inception4b, count, names, modules, depths)
    count+=2
    names, modules, depths = add_module('inception4c', model.inception4c, count, names, modules, depths)
    count+=2
    names, modules, depths = add_module('inception4d', model.inception4d, count, names, modules, depths)
    count+=2
    #names, modules, depths = add_module('inception4e', model.inception4e, count, names, modules, depths)
    names, modules, depths = add_module('maxpool4', model.maxpool4, count, names, modules, depths)

    count+=2
    names, modules, depths = add_module('inception5a', model.inception5a, count, names, modules, depths)
    count+=2
    #names, modules, depths = add_module('inception5b', model.inception5b, count, names, modules, depths)

    #Linear classifier
    names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
    #names, modules, depths = add_module('dropout', model.dropout, count, names, modules, depths)
    count+=1
    names, modules, depths = add_module('fc', model.fc, count, names, modules, depths)

    return modules, names, depths

#*******************************************************************************
def getdensenet(model, red=1):
    def add_submodules(block, count, red, names, modules, depths, transition = True):
        count0=count
        nsubm = len(list(block.children()))

        for i in range(red, nsubm, red):
            count+=red*2
            module = list(block.children())[i]
            names, modules, depths = add_module(module.__class__.__name__, module, count, names, modules, depths)

        if transition: count = count0+nsubm*2+1
        else: count = count0+nsubm*2

        return names, modules, depths, count

    count = 0
    modules = []
    names = []
    depths = []

    names, modules, depths = add_module('input', 'input', 0, names, modules, depths)

    count+=1
    names, modules, depths = add_module('maxpool', model.features.pool0, count, names, modules, depths)

    names, modules, depths, count = add_submodules(model.features.denseblock1, count, red, names, modules, depths)
    names, modules, depths = add_module('denseblock1', model.features.transition1, count, names, modules, depths)

    names, modules, depths, count = add_submodules(model.features.denseblock2, count, red, names, modules, depths)
    names, modules, depths = add_module('denseblock2', model.features.transition2, count, names, modules, depths)

    names, modules, depths, count = add_submodules(model.features.denseblock3, count, red, names, modules, depths)
    names, modules, depths = add_module('denseblock3', model.features.transition3, count, names, modules, depths)

    names, modules, depths, count = add_submodules(model.features.denseblock4, count, red, names, modules, depths, transition = False)
    names, modules, depths = add_module('denseblock4', model.features.norm5, count, names, modules, depths)

    count+=1
    names, modules, depths = add_module('fc', model.classifier, count, names, modules, depths)

    return modules, names, depths

#*******************************************************************************
def get_fully_connected_biroli_depths(model):
    count = 0
    modules = []
    names = []
    depths = []
    names, modules, depths = add_module('input', 'input', 0, names, modules, depths)
    count+=1

    for i, module in enumerate(model.children()):
        name = module.__class__.__name__
        if 'Sequential' in name:
            for i, submodule in enumerate(module.children()):
                name = submodule.__class__.__name__
                names, modules, depths = add_module(name, submodule, count, names, modules, depths)
                # if 'Linear' in name:
                #     names, modules, depths = add_module(name, submodule, count, names, modules, depths)
                #     count+=1
                # if 'BatchNorm1d' in name:
                #     names, modules, depths = add_module(name, submodule, count, names, modules, depths)
                #     count+=1
        else:
            #print(name)
            names, modules, depths = add_module(name, module, count, names, modules, depths)
    return modules, names, depths


#*******************************************************************************
def getquickc10(model):
    count = 0

    modules = []
    names = []
    depths = []

    names, modules, depths = add_module('input', 'input', 0, names, modules, depths)
    count+=1
    names, modules, depths = add_module('conv1', model.conv1, count, names, modules, depths)
    count+=1
    names, modules, depths = add_module('conv2', model.conv2, count, names, modules, depths)
    count+=1
    names, modules, depths = add_module('conv3', model.conv3, count, names, modules, depths)
    names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
    count+=1
    names, modules, depths = add_module('fc', model.classifier, count, names, modules, depths)

    return modules, names, depths

#*******************************************************************************
#*******************************************************************************


# from my_nets.wide_resnet import wide_resnet28, wide_resnet16, wide_resnet10, wide_resnet40
# from torchsummary import summary
# from torch import nn
# net = wide_resnet16().to('cuda')
# summary(net,(3,32, 32), device = 'cuda')
# net.modules()

# def add_submodules_(block, count, red, names, modules, depths):
#     for i, module in enumerate(block):
#         count += update_conv_count(module, 0)
#         if count in depths: continue
#         if (i+1)%red==0:
#             names, modules, depths = add_module(module.__class__.__name__, module, count, names, modules, depths)
#     return names, modules, depths, count

# def getWideResNetDepths_(model, red):
#
#     modules = []
#     names = []
#     depths = []
#
#     count = 0
#     names, modules, depths = add_module('input', 'input', count, names, modules, depths)
#     count += 1
#     names, modules, depths = add_module('conv1', model.conv0, count, names, modules, depths)
#
#     count_block = count + getLayerDepth(model.group0)
#     names, modules, depths = add_module('gropu0', model.group0, count_block, names, modules, depths)
#     names, modules, depths, count = add_submodules(model.group0, count, red, names, modules, depths)
#
#     count_block = count + getLayerDepth(model.group1)
#     names, modules, depths = add_module('group1', model.group1, count_block, names, modules, depths)
#     names, modules, depths, count = add_submodules(model.group1, count, red, names, modules, depths)
#
#     count_block = count + getLayerDepth(model.group2)
#     names, modules, depths = add_module('group2', model.group2, count_block, names, modules, depths)
#     names, modules, depths, count = add_submodules(model.group2, count, red, names, modules, depths)
#
#     names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
#     count += 1
#     names, modules, depths = add_module('fc', model.fc, count, names, modules, depths)
#
#     names = [x for _, x in sorted(zip(depths,names), key=lambda x: x[0])]
#     modules = [x for _, x in sorted(zip(depths,modules), key=lambda x: x[0])]
#     depths = [x for _, x in sorted(zip(depths,depths), key=lambda x: x[0])]
#
#     return modules, names, depths


# def getResNetsDepths(model, red):
#     def add_submodules(block, count, red, names, modules, depths):
#         for i,module in enumerate(block):
#             count += get_submoduleDepth(module)
#             if count in depths:
#                 continue
#             if (i+1)%red==0:
#                 names, modules, depths = add_module(module.__class__.__name__, module, count, names, modules, depths)
#         return names, modules, depths, count
#
#     def get_submoduleDepth(layer):
#         count = 0
#         for c in layer.children():
#             name = c.__class__.__name__
#             if 'Conv' in name:
#                 count += 1
#         return count
#
#     modules = []
#     names = []
#     depths = []
#
#     # input
#     count = 0
#     names, modules, depths = add_module('input', 'input', count, names, modules, depths)
#
#     # maxpooling
#     count += 1
#     names, modules, depths = add_module('maxpool', model.maxpool, count, names, modules, depths)
#
#     #add output block1
#     count_block = count
#     count_block += getLayerDepth(model.layer1)
#     names, modules, depths = add_module('layer1', model.layer1, count_block, names, modules, depths)
#     #add sublayers block1
#     names, modules, depths, count = add_submodules(model.layer1, count, red, names, modules, depths)
#     #
#     count_block = count
#     count_block += getLayerDepth(model.layer2)
#     names, modules, depths = add_module('layer2', model.layer2, count_block, names, modules, depths)
#     names, modules, depths, count = add_submodules(model.layer2, count, red, names, modules, depths)
#
#     count_block = count
#     count_block += getLayerDepth(model.layer3)
#     names, modules, depths = add_module('layer3', model.layer3, count_block, names, modules, depths)
#     names, modules, depths, count = add_submodules(model.layer3, count, red, names, modules, depths)
#
#     if hasattr(model, "layer4"):  #in an example i removed the layer 4
#         count_block = count
#         count_block += getLayerDepth(model.layer4)
#         names, modules, depths = add_module('layer4', model.layer4, count_block, names, modules, depths)
#         names, modules, depths, count = add_submodules(model.layer4, count, red, names, modules, depths)
#
#     # average pooling
#     count += 1
#     names, modules, depths = add_module('avgpool', model.avgpool, count, names, modules, depths)
#
#     # output
#     count += 1
#     names, modules, depths = add_module('fc', model.fc, count, names, modules, depths)
#
#     #sort layers by depths
#     names = [x for _,x in sorted(zip(depths,names))]
#     modules = [x for _,x in sorted(zip(depths,modules))]
#     depths = [x for _,x in sorted(zip(depths,depths))]
#
#     return modules, names, depths



















# def getdensenet(model, layers, red=1):
#     def add_submodules(block, count, red, names, modules, depths):
#         nsubm = len(list(block.children()))
#         for i in range(0, nsubm, red):
#             if i==0:
#                 continue
#             else:
#                 count+=red*2
#             module = list(block.children())[i]
#             names, modules, depths = add_module(module.__class__.__name__, module, count, names, modules, depths)
#         return names, modules, depths
#
#     count = 0
#     modules = []
#     names = []
#     depths = []
#
#     names, modules, depths = add_module('input', 'input', 0, names, modules, depths)
#
#     count+=1
#     names, modules, depths = add_module('maxpool', model.features.pool0, count, names, modules, depths)
#
#     names, modules, depths= add_submodules(model.features.denseblock1, count, red, names, modules, depths)
#     count=layers[0]*2+2
#     names, modules, depths = add_module('denseblock1', model.features.transition1, count, names, modules, depths)
#
#     names, modules, depths = add_submodules(model.features.denseblock2, count, red, names, modules, depths)
#     count=(layers[1]+layers[0])*2+3
#     names, modules, depths = add_module('denseblock2', model.features.transition2, count, names, modules, depths)
#
#     names, modules, depths = add_submodules(model.features.denseblock3, count, red, names, modules, depths)
#     count=(layers[1]+layers[0]+layers[2])*2+4
#     names, modules, depths = add_module('denseblock3', model.features.transition3, count, names, modules, depths)
#
#     names, modules, depths = add_submodules(model.features.denseblock4, count, red, names, modules, depths)
#     count=(layers[1]+layers[0]+layers[2]+layers[3])*2+4
#     names, modules, depths = add_module('denseblock4', model.features.norm5, count, names, modules, depths)
#
#     count+=1
#     names, modules, depths = add_module('fc', model.classifier, count, names, modules, depths)
#
#     return modules, names, depths


# from torchvision.models import vgg11, resnet34, googlenet
#
#
# model = googlenet(pretrained=True)
# model = model.to('cuda')
# model.eval()
#
#
# print('Training mode : {}'.format(model.training))
# #selecting layers from which representation will be exctracted
# modules, names, depths= getgooglenetdepths(model)
# print('Number of layers : {}'.format(len(modules)))
# print('Layer names : {}'.format(names))
# print('Layer depths : {}'.format(list(depths)))
