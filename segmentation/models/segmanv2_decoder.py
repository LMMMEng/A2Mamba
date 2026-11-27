'''
This is an official implementation of the SegMAN-V2 decoder proposed in the paper:
https://arxiv.org/abs/2507.16624

The decoder aggregates multi-level backbone features and processes the fused feature
with three parallel branches (global average pooling, identity mapping, and the
MM-Refine module with an A2SSM-based MASS mixer), followed by a low-level feature
enhancement stage.
'''
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, to_2tuple
from mmdet.models.backbones.csp_darknet import Focus
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.segformer_head import resize, BaseDecodeHead

import selective_scan_cuda_oflex_rh


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


class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, lag=0):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)


class LayerScale(nn.Module):

    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()

        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias

        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class A2SSM(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=1,
        d_conv=5,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0,
        device=None,
        dtype=None,
        deploy=False,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.conv2d = nn.Sequential(
            DilatedReparamBlock(self.d_inner, d_conv, deploy=deploy),
            nn.BatchNorm2d(self.d_inner),
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def _selective_scan_state_fn(self, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)

    def forward_ssm(self, x, state_params=None):

        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        state = self._selective_scan_state_fn(xs, dts, As, Bs, None, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False)
        state = rearrange(state, "b d s (h w) -> b (d s) (h w)", h=H, w=W)

        if state_params is not None:
            num_heads = state_params.shape[1]
            state = rearrange(state, 'b (n c) l -> b n l c', n=num_heads)
            state = einsum(state_params, state, 'b n l m, b n m c -> b n l c')
            state = rearrange(state, 'b n (h w) c -> b (n c) (h w)', h=H, w=W)

        y = state * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x, state_params=None):

        B, C, H, W = x.shape

        x = self.act(self.conv2d(x))
        y = self.forward_ssm(x, state_params=state_params)

        y = rearrange(y, 'b d (h w) -> b d h w', h=H, w=W)
        if self.dropout is not None:
            y = self.dropout(y)

        return y


class FFN(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ):
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):

        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MASS(nn.Module):

    def __init__(self,
                 embed_dim=64,
                 num_heads=8,
                 drop_path=0,
                 layerscale=False,
                 layer_init_values=1e-6,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d,
                 deploy=False):
        super().__init__()
        self.layerscale = layerscale
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.ssm = A2SSM(d_model=embed_dim, d_conv=5, deploy=deploy)

        self.lepe = nn.Sequential(
            DilatedReparamBlock(embed_dim, kernel_size=5, deploy=deploy),
            nn.BatchNorm2d(embed_dim),
        )

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward_mixer(self, x):

        B, C, H, W = x.shape

        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, 2*C:, ...])
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n (h w) c', m=3, n=self.num_heads)
        attn = einsum(q, k, 'b n l c, b n m c -> b n l m') * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = einsum(attn, v, 'b n l m, b n m c -> b n l c')
        x = rearrange(x, 'b n (h w) c -> b (n c) h w', h=H, w=W)
        x = x + self.ssm(x, state_params=attn)
        x = x + lepe
        x = self.proj(x)

        return x

    def forward(self, x):

        x = x + self.cpe1(x)
        x = x + self.drop_path(self.layer_scale1(self.forward_mixer(self.norm1(x))))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))

        return x


@HEADS.register_module()
class SegMANV2Decoder(BaseDecodeHead):

    def __init__(self,
                 inner_dim=128,
                 use_low_level_feat=False,
                 low_level_proj_dim=16,
                 final_proj_dim=160,
                 head_dim=32,
                 deploy=False,
                 drop_path=0.1,
                 **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)

        del self.conv_seg

        embed_dim = self.channels
        mixer_dim = embed_dim * 3
        fusion_dim = embed_dim * 3 + mixer_dim
        self.use_low_level_feat = use_low_level_feat

        self.linear_c4 = nn.Conv2d(self.in_channels[3], inner_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(self.in_channels[2], inner_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(self.in_channels[1], inner_dim, kernel_size=1)

        if use_low_level_feat:
            self.linear_c1 = nn.Conv2d(self.in_channels[0], low_level_proj_dim, kernel_size=1)
        else:
            self.linear_c1 = nn.Identity()

        self.linear_fuse = ConvModule(inner_dim*3,
                                      embed_dim,
                                      kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))


        self.identity = ConvModule(embed_dim, embed_dim, kernel_size=1,
                                   norm_cfg=dict(type='BN', requires_grad=True),
                                   act_cfg=None)

        self.local_proj = nn.Sequential(
            DilatedReparamBlock(embed_dim, kernel_size=5, deploy=deploy),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim,
                       embed_dim,
                       kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=None),
            )

        self.patch_embed1 = ConvModule(embed_dim,
                                       embed_dim,
                                       kernel_size=3,
                                       padding=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.patch_embed2 = ConvModule(embed_dim,
                                       embed_dim*2,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.patch_embed3 = ConvModule(embed_dim*2,
                                       embed_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.focus1 = Focus(embed_dim,
                            embed_dim,
                            kernel_size=3,
                            stride=2,
                            norm_cfg=dict(type='BN'),
                            act_cfg=None)

        self.focus2 = Focus(embed_dim*2,
                            embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='BN'),
                            act_cfg=None)

        self.mixer = MASS(embed_dim=mixer_dim, num_heads=mixer_dim // head_dim, deploy=deploy, drop_path=drop_path)

        self.fusion = ConvModule(fusion_dim,
                                 final_proj_dim,
                                 kernel_size=1,
                                 norm_cfg=dict(type='BN', requires_grad=True))

        if use_low_level_feat:
            fusion_dim = final_proj_dim + low_level_proj_dim
            self.out_proj = nn.Sequential(
                nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1, groups=fusion_dim),
                ConvModule(fusion_dim,
                           final_proj_dim,
                           kernel_size=1,
                           norm_cfg=dict(type='BN', requires_grad=True)),
            )


        self.conv_seg = nn.Conv2d(final_proj_dim, self.num_classes, kernel_size=1)

        self._convert_sync_batchnorm()

    def _convert_sync_batchnorm(self):
        if torch.distributed.is_initialized():
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward_mlp_decoder(self, inputs):

        c1, c2, c3, c4 = inputs

        H, W = c2.shape[2:]

        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)

        c4 = resize(c4, size=(H, W), mode='bilinear', align_corners=False)
        c3 = resize(c3, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.cat([c2, c3, c4], dim=1)
        if self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.linear_fuse(x)

        return (x, c1)

    def forward_ssm(self, x):

        B, C, H, W = x.shape

        identity = self.identity(x)

        x = x + self.patch_embed1(x)

        pool = self.image_pool(x).expand_as(x)
        local = self.local_proj(x)

        x1 = self.focus1(x)
        x2 = self.patch_embed2(x)
        x3 = self.patch_embed3(x2)
        x2 = self.focus2(x2)

        x = torch.cat([x1, x2, x3], dim=1)

        if self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.mixer(x)

        x = resize(x, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.cat([x, local, pool, identity], dim=1)
        if self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.fusion(x)

        return x

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        x, c1 = self.forward_mlp_decoder(x)
        x = self.forward_ssm(x)

        if self.use_low_level_feat:

            x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, c1], dim=1)
            if self.dropout_ratio > 0:
                x = self.dropout(x)
            x = self.out_proj(x)

        x = self.cls_seg(x)

        return x


if __name__ == "__main__":

    device = 'cuda'
    size = (2048, 1024)

    x = [
        torch.randn(2, 64, size[0]//4, size[1]//4).to(device),
        torch.randn(2, 128, size[0]//8, size[1]//8).to(device),
        torch.randn(2, 256, size[0]//16, size[1]//16).to(device),
        torch.randn(2, 512, size[0]//32, size[1]//32).to(device),
    ]

    model = SegMANV2Decoder(in_channels=[64, 128, 256, 512],
                            in_index=[0, 1, 2, 3],
                            channels=128,
                            inner_dim=288,
                            head_dim=32,
                            num_classes=19)
    model.to(device)
    print(model)
    y = model(x)
    print(y.shape)

    loss_fn = nn.CrossEntropyLoss()
    label = torch.rand(2, size[0]//8, size[1]//8).random_(2).long().to(device)

    loss = loss_fn(y, label)
    loss.backward()
