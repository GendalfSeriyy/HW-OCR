import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):

    def __init__(self, nc, *args, **kwargs):
        super(FCNN, self).__init__()


        cnn = nn.Sequential()
        one_d_cnn = nn.Sequential()

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
        
        one_d_cnn.add_module('conv1d_1',
                           nn.Conv1d(512, 256, 5, 1, 1))
        one_d_cnn.add_module('batchnorm_1', nn.BatchNorm1d(256))
        one_d_cnn.add_module('relu_1', nn.ReLU(True))
        one_d_cnn.add_module('conv1d_2',
                           nn.Conv1d(256, 256, 3, 1, 1, 2))
        one_d_cnn.add_module('batchnorm_2', nn.BatchNorm1d(256))
        one_d_cnn.add_module('relu_2', nn.ReLU(True))
        one_d_cnn.add_module('conv1d_3',
                           nn.Conv1d(256, 128, 3, 1, 1, 3))
        one_d_cnn.add_module('batchnorm_3', nn.BatchNorm1d(128))
        one_d_cnn.add_module('relu_3', nn.ReLU(True))
        one_d_cnn.add_module('conv1d_4',
                           nn.Conv1d(128, 128, 3, 1, 1))
        one_d_cnn.add_module('batchnorm_4', nn.BatchNorm1d(128))
        one_d_cnn.add_module('relu_4', nn.ReLU(True))
        one_d_cnn.add_module('conv1d_5',
                           nn.Conv1d(128, 80, 1, 1, 0))
        

        self.cnn = cnn
        self.one_d_cnn = one_d_cnn

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        conv = self.one_d_cnn(conv.squeeze(2))
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        # add log_softmax to converge output
        output = F.log_softmax(conv, dim=2)

        return output


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero