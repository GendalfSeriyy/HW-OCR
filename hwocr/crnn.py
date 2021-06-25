import torch.nn as nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import math
import pandas as pd
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import CTCLoss


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
    
class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, *args, **kwargs):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        cnn = nn.Sequential()

        def ConvBlock(in_c, out_c, k_s, padding, stride, dilation = 1, batchNorm=False, dropout = None, pooling = None, number = 0):
            
            cnn.add_module('conv{0}'.format(number),
                           nn.Conv2d(in_c, out_c, k_s, stride, padding, dilation))
            if batchNorm:
                cnn.add_module('batchnorm{0}'.format(number), nn.BatchNorm2d(out_c))
            cnn.add_module('relu{0}'.format(number), nn.ReLU(True))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(number), nn.Dropout2d(p=dropout))
            if pooling is not None:
                cnn.add_module('pooling{0}'.format(number), nn.MaxPool2d(*pooling))
        
        ConvBlock(in_c=nc, out_c=16, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = None, pooling = [2,2], number=0)
        ConvBlock(in_c=16, out_c=32, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = 0.2, pooling = [2,2], number=1)
        ConvBlock(in_c=32, out_c=64, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = 0.2, pooling = None, number=2)
        ConvBlock(in_c=64, out_c=128, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = None, pooling = [(2, 2), (2, 1), (0, 1)], number=3)
        ConvBlock(in_c=128, out_c=128, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = 0.1, pooling = None, number=4)
        ConvBlock(in_c=128, out_c=128, k_s=3, padding=1, stride=1, 
                  dilation = 1, batchNorm=True, dropout = 0.1, pooling = [(2, 2), (2, 1), (0, 1)], number=5)
        ConvBlock(in_c=128, out_c=256, k_s=3, padding=0, stride=(1,2), 
                  dilation = 1, batchNorm=True, dropout = None, pooling = None, number=6)
        ConvBlock(in_c=256, out_c=512, k_s=2, padding=0, stride=(2,1), 
                  dilation = 1, batchNorm=True, dropout = None, pooling = None, number=7)
        
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        
        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero