import torch.nn as nn
import numpy as np

class fully_connected_new(nn.Module):

    def __init__(self, widths, depth = 6, input_size = 10, output_size = 1, orthog_init = False,
            dropout = False, batch_norm = True, p =0.5):
        super(fully_connected_new, self).__init__()

        if isinstance(widths, int):
            widths = np.array([input_size]+[widths for i in range(depth -1)], dtype = 'int')
        elif isinstance(widths, list):
            widths = np.array([input_size]+widths, dtype = 'int')
        else: raise TypeError('expected type int or list of variable widths')

        self.linear_body = make_layers_new(widths, batch_norm, dropout, p=p)
        self.linear_out = nn.Linear(widths[-1], output_size)

        if orthog_init: self.orthog_init()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                #nn.init.constant_(m.weight, 1) in resnet: (and by default!!!!)
                #incredible increase in performance if m.weight is set to random uniform 0,1 from constant 1
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)
                nn.init.zeros_(m.running_mean)
                nn.init.ones_(m.running_var)
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight) #by default is kaiming_uniform
                #nn.init.zeros_(m.bias) #by default is uniform(fan_in**0.5)

    def orthog_init(self):
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linear_body(x)
        x = self.linear_out(x)
        return x

def make_layers_new(widths, batch_norm, dropout, p):

    layers = []

    if dropout and batch_norm:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.BatchNorm1d(widths[i+1]), nn.ReLU(inplace = True), nn.Dropout(p=p),])

    elif dropout and not batch_norm:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True), nn.Dropout(p = p)])

    elif batch_norm and not dropout:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.BatchNorm1d(widths[i+1]), nn.ReLU(inplace = True) ])
    else:
        for i in range(len(widths)-1):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True)])

    return nn.Sequential(*layers)

#-------------------------------------------------------------------------------
 #from the paper
# Scaling description of generalization with number of parameters.
# 2020 Interdisciplinary Stat Mech

#original network Geiger et al
class fully_connected_old(nn.Module):

    def __init__(self, width, depth = 6, input_size = 10, output_size = 1, orthog_init = True,
            dropout = False, bottleneck = None, layer_bottleneck = 5):
        super(fully_connected_old, self).__init__()
        assert layer_bottleneck < depth

        bottleneck = width if bottleneck is None else bottleneck
        assert bottleneck<=width

        win = bottleneck if layer_bottleneck == 1 else width
        wout = bottleneck if layer_bottleneck == (depth-1) else width

        if dropout: self.linear_in = nn.Sequential(nn.Linear(input_size, win), nn.ReLU(inplace=True), nn.Dropout())
        else: self.linear_in = nn.Sequential(nn.Linear(input_size, win),  nn.ReLU(inplace=True))

        self.linear_body = make_layers_old(depth-2, width, dropout, layer_bottleneck, bottleneck)
        self.linear_out = nn.Linear(wout, output_size)

        if orthog_init: self.orthog_init()

    def orthog_init(self): self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.linear_body(x)
        x = self.linear_out(x)
        return x


def make_layers_old(block_depth, width, dropout, layer_bottleneck, bottleneck):

    widths = width*np.ones(block_depth+1, dtype = int)
    widths[layer_bottleneck-1] = bottleneck
    layers = []

    if dropout:
        for i in range(block_depth):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True), nn.Dropout(),])
    else:
        for i in range(block_depth):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(inplace = True)])

    return nn.Sequential(*layers)
