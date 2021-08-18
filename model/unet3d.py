##From https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/unet3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        in_chan, out_chan = int(in_chan), int(out_chan)
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act, deepest_layer=False):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth
        self.deepest_layer = deepest_layer

    def forward(self, x):
        if self.deepest_layer:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out
    

class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #out = self.sigmoid(self.final_conv(x))
        out = self.final_conv(x)
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1,out_channels=1, act='relu'):
        super(UNet3D, self).__init__()
        
        self.down_tr64 = DownTransition(in_channels,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act,deepest_layer=True)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, out_channels)

    def forward(self, x):
        '''
        x = self.conv_init(x)
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        '''
        # I changed the forward function to make it more memory-friendly
        x, skip_out64 = self.down_tr64(x);
        x, skip_out128 = self.down_tr128(x);
        x, skip_out256 = self.down_tr256(x);
        x, skip_out512 = self.down_tr512(x);
        del skip_out512;

        x = self.up_tr256(x, skip_out256);
        del skip_out256;
        x = self.up_tr128(x, skip_out128);
        del skip_out128;
        torch.cuda.reset_max_memory_allocated();
        torch.cuda.empty_cache();
        x = self.up_tr64(x, skip_out64);
        del skip_out64;
        out = self.out_tr(x);
        return out
        
        
class UNet3D_small(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1,out_channels=1, act='relu'):
        super(UNet3D_small, self).__init__()
        
        self.down_tr64 = DownTransition(in_channels,-1,act)
        self.down_tr128 = DownTransition(32,0,act)
        self.down_tr256 = DownTransition(64,1,act,deepest_layer=True)

        self.up_tr128 = UpTransition(128,128, 0,act)
        self.up_tr64 = UpTransition(64,64,-1,act)
        self.out_tr = OutputTransition(32, out_channels)
        

    def forward(self, x):
        x, skip_out64 = self.down_tr64(x);
        x, skip_out128 = self.down_tr128(x);
        x, skip_out256 = self.down_tr256(x);
        del skip_out256;

        x = self.up_tr128(x, skip_out128);
        del skip_out128;
        x = self.up_tr64(x, skip_out64);
        del skip_out64;
        out = self.out_tr(x);
        return out
        
        
class UNet3D_do(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, act='relu'):
        super(UNet3D_do, self).__init__()
        
        self.down_tr64 = DownTransition(in_channels,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.do1 = nn.Dropout3d(p=0.5)
        self.down_tr256 = DownTransition(128,2,act)
        self.do2 = nn.Dropout3d(p=0.5)
        self.down_tr512 = DownTransition(256,3,act)

        self.do3 = nn.Dropout3d(p=0.5)
        self.up_tr256 = UpTransition(512, 512,2,act)
        self.do4 = nn.Dropout3d(p=0.5)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, out_channels)
    
    def forward(self, x):
        # I changed the forward function to make it more memory-friendly
        x, skip_out64 = self.down_tr64(x);
        x, skip_out128 = self.down_tr128(x);
        x = self.do1(x)
        x, skip_out256 = self.down_tr256(x);
        x = self.do2(x)
        x, skip_out512 = self.down_tr512(x);
        del skip_out512;

        x = self.do3(x)
        x = self.up_tr256(x, skip_out256);
        x = self.do4(x)
        del skip_out256;
        x = self.up_tr128(x, skip_out128);
        del skip_out128;
        torch.cuda.reset_max_memory_allocated();
        torch.cuda.empty_cache();
        x = self.up_tr64(x, skip_out64);
        del skip_out64;
        out = self.out_tr(x);
        
        return out
        
        
class UNet3D_sel(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, act='relu'):
        super(UNet3D_sel, self).__init__()
        
        self.down_tr64 = DownTransition(in_channels, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr_seg = OutputTransition(64, out_channels)
        self.out_tr_aux = OutputTransition(64, out_channels)
        
        self.out_tr_sel = nn.Sequential(LUConv(64, 8, act),
                                        LUConv(8, 1, act))
                                        
    def forward(self, x):
            x, skip_out64 = self.down_tr64(x);
            x, skip_out128 = self.down_tr128(x);
            x, skip_out256 = self.down_tr256(x);
            x, skip_out512 = self.down_tr512(x);
            del skip_out512;
    
            x = self.up_tr256(x, skip_out256);
            del skip_out256;
            x = self.up_tr128(x, skip_out128);
            del skip_out128;
            torch.cuda.reset_max_memory_allocated();
            torch.cuda.empty_cache();
            x = self.up_tr64(x, skip_out64);
            del skip_out64;
            out_seg = self.out_tr_seg(x);
            out_aux = self.out_tr_aux(x);
            out_sel = self.out_tr_sel(x);
            
            out = {}
            out['seg'] = out_seg
            out['aux'] = out_aux
            out['sel'] = out_sel
            return out
            
        
class UNet3D_aux(nn.Module):
    def __init__(self, in_channels=1,out_channels=1, aux_channels=1, act='relu'):
        super(UNet3D_aux, self).__init__()
        self.down_tr64 = DownTransition(in_channels,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act,deepest_layer=True)

        self.up_tr256 = UpTransition(512, 512,2,act)
        
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr128_aux = UpTransition(256,256, 1,act)

        self.up_tr64 = UpTransition(128,128,0,act)
        self.up_tr64_aux = UpTransition(128,128,0,act)
        self.up_tr64_last_conv_aux = LUConv(64, 64, act)
        
        self.out_tr = OutputTransition(64, out_channels)
        self.out_tr_aux = OutputTransition(64, aux_channels)

    def forward(self, x):
        x, skip_out64 = self.down_tr64(x);
        x, skip_out128 = self.down_tr128(x);
        x, skip_out256 = self.down_tr256(x);
        x, skip_out512 = self.down_tr512(x);
        del skip_out512;

        x = self.up_tr256(x, skip_out256);
        del skip_out256;
        x = self.up_tr128(x, skip_out128);
        del skip_out128;
        torch.cuda.reset_max_memory_allocated();
        torch.cuda.empty_cache();
        x_seg = self.up_tr64(x, skip_out64);
        
        ### up_tr64 ###
        out_up_conv = self.up_tr64.up_conv(x)
        concat = torch.cat((out_up_conv, skip_out64),1)
        out = self.up_tr64.ops[0](concat)
        x = self.up_tr64.ops[1](out)
        x_aux = self.up_tr64_last_conv_aux(out)
        ###
        
        del skip_out64;
        out_seg = self.out_tr(x_seg);
        out_aux = self.out_tr_aux(x_aux);
    
        out = {}
        out['seg'] = out_seg
        out['aux'] = out_aux
        return out
    