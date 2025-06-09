import torch
from torch import nn


class concat_module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=False):
        super(concat_module, self).__init__()
        assert dimension in [1, 2, 3]
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.bn_layer = bn_layer

        

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            if self.sub_sample:
                max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            if self.sub_sample:
                max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if self.sub_sample:
                max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d


        self.conv1 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.conv2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)


    def forward(self, input_x, input_y):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        x = self.conv1(input_x)
        y = self.conv2(input_y)

        z = torch.cat((x, y), dim=1)

        if self.bn_layer:
            z = self.bn(z)
        # res link
        out = z + input_x

        return out


class concat_module1D(concat_module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=False):
        super(concat_module1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample, bn_layer=False)


class concat_module2D(concat_module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=False):
        super(concat_module2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample, bn_layer=False)


class concat_module3D(concat_module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=False):
        super(concat_module3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample, bn_layer=False)


if __name__ == '__main__':
    import torch

    rgb_img = torch.randn(32, 64, 48, 14, 14)
    heatmap_img = torch.randn(32, 64, 48, 14, 14)
    net = concat_module3D(64)
    out = net(rgb_img, heatmap_img)
    print(out.size()) 



