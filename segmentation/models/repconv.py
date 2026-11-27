import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple



def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std


def add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.size(2)
    asym_w = asym_kernel.size(3)
    square_h = square_kernel.size(2)
    square_w = square_kernel.size(3)
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h, 
                  square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel
    return square_kernel


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)
    

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)

    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    
    return merged_kernel


def merge_dilated_into_large_kernel_strip(large_kernel, dilated_kernel, dilated_r):
    
    large_k = large_kernel.shape[2:]
    dilated_k = dilated_kernel.shape[2:]
    identity_kernel = torch.zeros_like(large_kernel).to(large_kernel.device)
    
    if large_k[0] == 1:
        lk = large_k[1]
        dk = dilated_k[1]
        dr = dilated_r[1]
        
        if dr > 1:
            identity_kernel[..., 0::dr] = dilated_kernel
        else:
            padding = (lk - dk) // 2  
            identity_kernel[..., padding:(padding+dk)] = dilated_kernel

    elif large_k[1] == 1:
        
        lk = large_k[0]
        dk = dilated_k[0]
        dr = dilated_r[0]
        
        if dr > 1:
            identity_kernel[..., 0::dr, :] = dilated_kernel
        else:
            padding = (lk - dk) // 2  
            identity_kernel[..., padding:(padding+dk), :] = dilated_kernel
    
    merged_kernel = large_kernel + identity_kernel

    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True, use_axial_conv=True, use_1x1_conv=False):
        super().__init__()
        
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        

        self.attempt_use_lk_impl = attempt_use_lk_impl
        self.use_axial_conv = use_axial_conv
        self.use_1x1_conv = use_1x1_conv

        if kernel_size == 7:
            self.kernel_sizes = [3, 3, 3, 5, 5]
            self.dilates = [1, 2, 3, 1, 2]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            
            self.origin_bn = get_bn(channels, use_sync_bn)
            
            if use_axial_conv:
        
                self.lk_origin_h = get_conv2d(channels, channels, kernel_size=(1, kernel_size), stride=1,
                                            padding=(0, kernel_size//2), dilation=1, groups=channels, bias=False,
                                            attempt_use_lk_impl=attempt_use_lk_impl)
                self.lk_origin_w = get_conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1,
                                            padding=(kernel_size//2, 0), dilation=1, groups=channels, bias=False,
                                            attempt_use_lk_impl=attempt_use_lk_impl)
            
                self.origin_bn_h = get_bn(channels, use_sync_bn)
                self.origin_bn_w = get_bn(channels, use_sync_bn)
            
            if use_1x1_conv:
                self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
                self.pw_bn = get_bn(channels, use_sync_bn)
                  
            for k, r in zip(self.kernel_sizes, self.dilates):
                
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))
                
                
                if use_axial_conv:
                    
                    axial_kernel = [(1, k), (k, 1)]
                    axial_padding = [(0, (r * (k - 1) + 1) // 2), ((r * (k - 1) + 1) // 2, 0)]
                    axial_dil = [(1, r), (r, 1)]
                    
                    for ak, ap, ad in zip(axial_kernel, axial_padding, axial_dil):

                        k = 'x'.join(map(str, ak))
                        r = 'x'.join(map(str, ad))
                        
                        self.__setattr__('ax_conv_k{}_d{}'.format(k, r),
                                    nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ak, stride=1,
                                            padding=ap, dilation=ad, groups=channels,
                                            bias=False))
                        self.__setattr__('ax_bn_k{}_d{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))
                

    def forward(self, x):
        
        if not hasattr(self, 'origin_bn'):   # deploy mode
            x = self.lk_origin(x)
            return x
        
        out = self.origin_bn(self.lk_origin(x))
        
        if self.use_axial_conv:

            out = out + self.origin_bn_h(self.lk_origin_h(x))
            out = out + self.origin_bn_w(self.lk_origin_w(x))
        
        if self.use_1x1_conv:
            out = self.pw_bn(self.pwconv(x)) + out
        
        for k, r in zip(self.kernel_sizes, self.dilates):
            
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
            
            if self.use_axial_conv:
                
                axial_kernel = [(1, k), (k, 1)]
                axial_dil = [(1, r), (r, 1)]
                
                for ak, ad in zip(axial_kernel, axial_dil):
                
                    k = 'x'.join(map(str, ak))
                    r = 'x'.join(map(str, ad))
                    
                    conv = self.__getattr__('ax_conv_k{}_d{}'.format(k, r))
                    bn = self.__getattr__('ax_bn_k{}_d{}'.format(k, r))
                    
                    out = out + bn(conv(x))
                    
        return out

    def merge_dilated_branches(self):
        # print('start deploy')
        if hasattr(self, 'origin_bn'):
            
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            
            if self.use_axial_conv:
                
                origin_kh, origin_bh = fuse_bn(self.lk_origin_h, self.origin_bn_h)
                origin_kw, origin_bw = fuse_bn(self.lk_origin_w, self.origin_bn_w)
                
                self.__delattr__('lk_origin_h')
                self.__delattr__('lk_origin_w')
                
                self.__delattr__('origin_bn_h')
                self.__delattr__('origin_bn_w')
            
            for k, r in zip(self.kernel_sizes, self.dilates):
                
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
                
                if self.use_axial_conv:
                    
                    axial_kernel = [(1, k), (k, 1)]
                    axial_dil = [(1, r), (r, 1)]
                    
                    for ak, ad in zip(axial_kernel, axial_dil):
                    
                        k = 'x'.join(map(str, ak))
                        r = 'x'.join(map(str, ad))
                        
                        conv = self.__getattr__('ax_conv_k{}_d{}'.format(k, r))
                        bn = self.__getattr__('ax_bn_k{}_d{}'.format(k, r))

                        branch_k, branch_b = fuse_bn(conv, bn)
                        
                        if ak[0] == 1:
                            origin_kh = merge_dilated_into_large_kernel_strip(origin_kh, branch_k, ad)
                            origin_bh += branch_b
                        else:
                            origin_kw = merge_dilated_into_large_kernel_strip(origin_kw, branch_k, ad)
                            origin_bw += branch_b
            
            if self.use_axial_conv:
                
                origin_k = add_to_square_kernel(origin_k, origin_kh)
                origin_k = add_to_square_kernel(origin_k, origin_kw)
                origin_b = origin_b + origin_bh + origin_bw

            if self.use_1x1_conv:
                
                pw_k, pw_b = fuse_bn(self.pwconv, self.pw_bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, pw_k, 1)
                origin_b += pw_b
            
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), 
                                     bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
                
                if self.use_axial_conv:
                    
                    axial_kernel = [(1, k), (k, 1)]
                    axial_dil = [(1, r), (r, 1)]
                    
                    for ak, ad in zip(axial_kernel, axial_dil):
                    
                        k = 'x'.join(map(str, ak))
                        r = 'x'.join(map(str, ad))
                       
                        self.__delattr__('ax_conv_k{}_d{}'.format(k, r))
                        self.__delattr__('ax_bn_k{}_d{}'.format(k, r))
