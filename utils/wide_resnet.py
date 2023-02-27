import torch
import torch.nn as nn
#*******************************************************************************
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=1, bias=False, dilation=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, block_type='pre_act', net_type = 'standard',
                                stride=1, downsample=None, pdrop = None, se_reduction = 8):
        super(BasicBlock, self).__init__()

        self.block_type = block_type
        self.net_type = net_type
        self.pdrop = pdrop

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        if pdrop is not None:
            self.dropout = nn.Dropout2d(p=pdrop) #p from paper wide resnet

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if block_type=='pyramid':
            self.bn3 = nn.BatchNorm2d(planes) #from pyramidal networks

        #squeeze excitation layer
        if net_type=='se':
            self.se = SELayer(planes, reduction=se_reduction) #from squeeze and excitation

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        if self.block_type=='pre_act':
            out = self.relu(out)

        out = self.conv1(out)
        if self.pdrop is not None:
            out = self.dropout(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.block_type=='pyramid':
            out = self.bn3(out)

        if self.net_type == 'se':
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

#BatchNorm + Relu is done before conv2d in standard resnet after....
class Wide_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, inplanes = 16, widening_factor = 1,
                                instride = 1, pdrop = None, block_type = 'pre_act', net_type = 'standard', se_reduction = 8):
        super(Wide_ResNet, self).__init__()

        #block type: pre_act or pyramid
        #net type: standard or se (squeeze and excitation, reduction factor = 8/4 to be tuned)
        self.inplanes = inplanes
        assert widening_factor*4*inplanes>num_classes
        assert block_type=='pre_act' or block_type=='pyramid'
        assert net_type=='standard' or net_type=='se'
        self.block_type = block_type
        self.net_type = net_type
        self.se_reduction = se_reduction



        k = widening_factor
        self.widths = [self.inplanes, self.inplanes*k, self.inplanes*2*k, self.inplanes*4*k]
        linear_activations = self.inplanes*4*k

        self.conv0 = nn.Conv2d(3, self.widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group0 = self._make_layer(block, self.widths[1], layers[0], stride = instride, pdrop = pdrop)
        self.group1 = self._make_layer(block, self.widths[2], layers[1], stride=2, pdrop = pdrop)
        self.group2 = self._make_layer(block, self.widths[3], layers[2], stride=2, pdrop = pdrop)
        self.bn = nn.BatchNorm2d(self.widths[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(linear_activations, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #in resnet mode = fan_out nonlinearity = 'relu'
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                #nn.init.constant_(m.weight, 1) in resnet: (and by default!!!!)
                #incredible increase in performance if m.weight is set to random uniform 0,1 from constant 1
                nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)
                nn.init.zeros_(m.running_mean)
                nn.init.ones_(m.running_var)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight) #by default is kaiming_uniform
                if m.bias is not None: nn.init.zeros_(m.bias) #by default is uniform(fan_in**0.5)

    def _make_layer(self, block, planes, blocks, stride, pdrop):
        downsample = None

        #downsampling is done only before layer3 and layer4 where stride is different from 1
        #in standard resnet also before layer2
        if stride != 1 or self.inplanes != planes:
            #in the downsampling there is no batchnorm nor bias
            #downsample = nn.Sequential(nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True), conv1x1(self.inplanes, planes, stride))
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride))

        layers = []
        layers.append(block(self.inplanes, planes, block_type = self.block_type, stride = stride,
                downsample = downsample, pdrop = pdrop, net_type = self.net_type, se_reduction = self.se_reduction))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,  block_type = self.block_type, pdrop = pdrop,
                                    net_type = self.net_type, se_reduction = self.se_reduction))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv0(x)
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(block, layers, **kwargs):
    model = Wide_ResNet(block, layers, **kwargs)
    return model

def wide_resnet10(**kwargs):
    return _resnet(BasicBlock, [1, 1, 1], **kwargs)

def wide_resnet16(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2], **kwargs)

def wide_resnet22(**kwargs):
    return _resnet(BasicBlock, [3, 3, 3], **kwargs)

def wide_resnet28(**kwargs):
    return _resnet(BasicBlock, [4, 4, 4], **kwargs)

def wide_resnet40(**kwargs):
    return _resnet(BasicBlock, [6, 6, 6], **kwargs)
