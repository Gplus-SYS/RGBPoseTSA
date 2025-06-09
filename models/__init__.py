from .cls_heads import *
from .resnet.resnet3d_slowonly import ResNet3dSlowOnly
from .I3D_backbone import I3D
from .I3D_TSA_baseline import backbone as baseline
from .I3D_TSA_V2 import backbone as I3D_TSA_V2
from .I3D_TSA_early_fusion import backbone as early_fusion
from .PS import PSNet
from .vit_decoder import decoder_fuser
from .MLP import MLP_score
