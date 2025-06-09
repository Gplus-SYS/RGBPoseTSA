import torch.nn as nn
from ..weight_init import kaiming_init,constant_init




class ConvModule(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True):
        super(ConvModule, self).__init__()

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace


        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        self.conv = None
        if conv_cfg['type'] == 'Conv3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        if self.with_norm:
            if self.norm_cfg['type'] == 'BN3d':
                norm = nn.BatchNorm3d(out_channels)
                self.norm_name = 'bn'
                self.add_module(self.norm_name, norm)
            # 可以根据需要添加其他归一化层
            elif self.norm_cfg['type'] == 'LN':
                norm = nn.LayerNorm([out_channels, 56, 56])
                self.norm_name = 'bn'
                self.add_module(self.norm_name, norm)
                
        if self.with_activation:
            if act_cfg['type'] == 'ReLU':
                self.activation = nn.ReLU(inplace=True)
            # 可以根据需要添加其他激活函数

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None
        
    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            if self.norm_cfg['type'] == 'LN':
                x = x.permute(0, 2, 1, 3, 4)
                x = self.norm(x)
                x = x.permute(0, 2, 1, 3, 4)
            else:
                x = self.norm(x)
        if self.with_activation:
            x = self.activation(x)
        return x