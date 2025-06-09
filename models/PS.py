import torch.nn as nn
from models.PS_parts import *


class PSNet(nn.Module):
    def __init__(self, frames=12, dropout=0):
        super(PSNet, self).__init__()
        self.dropout_ratio = dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.inc = inconv(frames, 12)
        self.down1 = down(12, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 96)
        self.down4 = down(96, 96) 
        self.tas = MLP_tas(64, 2)

    def forward(self, x):
        '''
        :param x: (batchsize, total_videos, channels)
        :return:
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.dropout is not None:
            x = self.dropout(x5)
        x = self.tas(x5)
        return x5, x
    