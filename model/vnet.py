#Corrected version of the implementation
#https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py


import torch
import torch.nn as nn
import torch.nn.functional as F

class vnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(vnet, self).__init__()

        if in_channels > 1:
            self.stage_0 = nn.Sequential(nn.Conv3d(3, 1, kernel_size=1, stride=1),
                                     nn.PReLU(1))
        else:
            self.stage_0 = nn.Sequential()
        
        self.stage_1_left_block = InitialConvBlock(1, 16, kernel_size=5, stride=1, padding=2)
        self.stage_1_down_conv = DownConvBlock(16, 32)
        
        self.stage_2_left_block = SkipConvBlock(32, 32, kernel_size=5, stride=1, padding=2, num_layers=2)
        self.stage_2_down_conv = DownConvBlock(32, 64)
        
        self.stage_3_left_block = SkipConvBlock(64, 64, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_3_down_conv = DownConvBlock(64, 128)
        
        self.stage_4_left_block = SkipConvBlock(128, 128, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_4_down_conv = DownConvBlock(128, 256)
        
        self.stage_5_left_block = SkipConvBlock(256, 256, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_5_up_conv = UpConvBlock(256, 128)
        
        self.stage_4_right_block = SkipConvBlock(256, 128, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_4_up_conv = UpConvBlock(128, 64)
        
        self.stage_3_right_block = SkipConvBlock(128, 64, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_3_up_conv = UpConvBlock(64, 32)
        
        self.stage_2_right_block = SkipConvBlock(64, 32, kernel_size=5, stride=1, padding=2, num_layers=3)
        self.stage_2_up_conv = UpConvBlock(32, 16)
        
        self.stage_1_right_block = SkipConvBlock(32, 16, kernel_size=5, stride=1, padding=2, num_layers=1)
        self.last_conv = nn.Sequential(nn.Conv3d(16, out_channels, kernel_size=1, stride=1),
                                       nn.PReLU(out_channels))


    def __call__(self, x):
        #import pdb
        #pdb.set_trace()
        x = self.stage_0(x);
        
        stage_1_out = self.stage_1_left_block(x);
        x = self.stage_1_down_conv(stage_1_out);

        stage_2_out = self.stage_2_left_block(x, x);
        x = self.stage_2_down_conv(stage_2_out);

        stage_3_out = self.stage_3_left_block(x, x);
        x = self.stage_3_down_conv(stage_3_out);

        stage_4_out = self.stage_4_left_block(x, x);
        #print(stage_4_out.shape)
        x = self.stage_4_down_conv(stage_4_out)
        #print(x.shape)
        
        x = self.stage_5_left_block(x, x)
        #print(x.shape)
        x = self.stage_5_up_conv(x)
        #print(x.shape)
        
        x = self.stage_4_right_block(torch.cat([stage_4_out, x], 1), x)
        #print(x.shape)
        x = self.stage_4_up_conv(x)
        #print(x.shape)
        
        x = self.stage_3_right_block(torch.cat([stage_3_out, x], 1), x)
        #print(x.shape)
        x = self.stage_3_up_conv(x)
        #print(x.shape)
        
        x = self.stage_2_right_block(torch.cat([stage_2_out, x], 1), x)
        #print(x.shape)
        x = self.stage_2_up_conv(x)
        #print(x.shape)
        
        x = self.stage_1_right_block(torch.cat([stage_1_out, x], 1), x)
        #print(x.shape)
        x = self.last_conv(x)
        #print(x.shape)

        return x


        
class SkipConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_layers):
        super(SkipConvBlock, self).__init__()
        self.block = make_conv_act_layers(in_channels, kernel_size, stride, padding, num_layers-1)
        self.transition_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                                             nn.PReLU(out_channels))
    def forward(self, x, x_skip):
        #x_combined = torch.cat([x, x_skip], 1)
        x = self.block(x)
        x = self.transition_conv(x)
        x_t = x.transpose(0,1)
        x_skip_t = x_skip.transpose(0,1)
        out_t = x_t + x_skip_t
        out = out_t.transpose(0,1)
        return out
        
        
class InitialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(InitialConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU(out_channels)
            )
            
    def forward(self, x):
        x_processed = self.block(x)
        x_processed_t = x_processed.transpose(0,1)
        x_t = x.transpose(0,1)
        out_t = x_processed_t + x_t
        out = out_t.transpose(0,1)
        return out

        
class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.PReLU(out_channels)
            )

    def forward(self, x):
        x = self.block(x)
        return x
        

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.PReLU(out_channels)
            )

    def forward(self, x):
        x = self.block(x)
        return x
        

def make_conv_act_layers(channels, kernel_size, stride, padding, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Conv3d(channels, channels, kernel_size, stride, padding))
        layers.append(nn.PReLU(channels))
    return nn.Sequential(*layers)