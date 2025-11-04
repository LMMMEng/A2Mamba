import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmengine.runner import load_checkpoint
from einops import einsum, rearrange, repeat
from natten.functional import na2d_av, na2d_qk
from natten.flops import qk_2d_rpb_flop, av_2d_flop
from fvcore.nn import flop_count, parameter_count
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.pvt_v2 import DropPath, to_2tuple, register_model


import selective_scan_cuda_oflex_rh



def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1 
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size

# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_C=True, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    if with_C:
        flops = 9 * B * L * D * N
    else:
        flops = 7 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


def selective_scan_state_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_C=False, with_D=False, with_Z=False)
    return flops


class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, 
                z=None, delta_bias=None, 
                delta_softplus=False,
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


def selective_scan_state_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                            return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)



class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        return (sin, cos)


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
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class A2SSM(nn.Module):
    '''
    Attention-augmented SSM layer
    '''
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
        num_heads=1,
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
        
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner)
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

        self.state_fusion = nn.Conv2d(num_heads, num_heads, kernel_size=1)
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

        state = selective_scan_state_fn(xs, dts, As, Bs, None, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False)
        state = rearrange(state, "b d s (h w) -> b (d s) (h w)", h=H, w=W)
        
        if state_params is not None:
            
            attn1, attn2, ws, dil = state_params
            num_heads = attn1.shape[1]
            
            state = rearrange(state, "b (m n d) (h w) -> m b n h w d", m=2, n=num_heads, h=H, w=W)
            state1 = na2d_av(attn1, state[0], kernel_size=toodd(ws))
            state2 = na2d_av(attn2, state[1], kernel_size=toodd(ws), dilation=dil)
            state = torch.cat([state1, state2], dim=1)
            state = rearrange(state, "b n h w d -> b n d (h w)")
            state = self.state_fusion(state).flatten(1, 2)

        with torch.cuda.amp.autocast(enabled=False):
            '''
            disable amp for better training stability
            '''
            state = state.to(torch.float32)
            Cs = Cs.to(torch.float32)
            x = state * Cs
            x = x + xs * Ds.view(-1, 1)

        return x
    

    def forward(self, x, state_params=None):
        
        B, C, H, W = x.shape

        x = self.act(self.conv2d(x))
        x = self.forward_ssm(x, state_params=state_params)

        x = rearrange(x, 'b d (h w) -> b d h w', h=H, w=W)
        if self.dropout is not None:
            x = self.dropout(x)

        return x



class Attention(nn.Module):
    def __init__(self, 
                 embed_dim=64, 
                 num_heads=2, 
                 window_size=7, 
                 window_dilation=1, 
                 global_mode=False, 
                 image_size=None, 
                 use_rpb=False, 
                 sr_ratio=1):
        
        super().__init__()
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation
        self.global_mode = global_mode
        self.sr_ratio = sr_ratio
        self.image_size = image_size
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
        
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.norm = LayerNorm2d(embed_dim)
        self.proj = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        
        if not global_mode:
            self.ssm = A2SSM(d_model=embed_dim, num_heads=num_heads)
            
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads//2, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)] * 2
            if global_mode: 
                rpb_list.append(nn.Parameter(torch.empty(1, num_heads, image_size[0]*image_size[1], image_size[0]*image_size[1]), requires_grad=True))
            self.rpb = nn.ParameterList(rpb_list)
            
        self.reset_parameters()
        

    def reset_parameters(self):
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item)
    

    def forward(self, x, pos_enc):
        
        B, C, H, W = x.shape
        dilation = [int(H/self.window_size[0]), int(W/self.window_size[1])]
        dilation = [min(dilation[0], self.window_size[0]), min(dilation[1], self.window_size[1])]
        
        gate = self.gate(x)
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])

        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)
        
        sin, cos = pos_enc
        q = theta_shift(q, sin, cos) * self.scale
        k = theta_shift(k, sin, cos)
        
        if hasattr(self, 'rpb'):
            rpb1, rpb2 = self.rpb[0], self.rpb[1]
        else:
            rpb1 = rpb2 = None
        
        q1, q2 = torch.chunk(q, chunks=2, dim=1)
        k1, k2 = torch.chunk(k, chunks=2, dim=1)
        v1, v2 = torch.chunk(v, chunks=2, dim=1)

        attn1 = na2d_qk(q1, k1, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb1)
        attn1 = torch.softmax(attn1, dim=-1)
        v1 = na2d_av(attn1, v1, kernel_size=toodd(self.window_size), dilation=self.window_dilation)       
        
        attn2 = na2d_qk(q2, k2, kernel_size=toodd(self.window_size), dilation=dilation, rpb=rpb2)
        attn2 = torch.softmax(attn2, dim=-1)
        v2 = na2d_av(attn2, v1+v2, kernel_size=toodd(self.window_size), dilation=dilation)
        
        x = torch.cat([v1, v2], dim=1)
        x = rearrange(x, 'b n h w c -> b (n c) h w')

        if not self.global_mode:
            state_params = [attn1, attn2, self.window_size, dilation]
            x = self.ssm(x, state_params=state_params)
        else:
            
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b (n c) h w -> b n (h w) c', n=self.num_heads, h=H, w=W)
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')
            
            if hasattr(self, 'rpb'):
                if self.rpb[-1].shape[2:] != attn.shape[2:]:
                    rpb = F.interpolate(self.rpb[-1], size=attn.shape[2:], mode='bicubic', align_corners=False)
                    attn = attn + rpb
                else:
                    attn = attn + self.rpb[-1]
                
            attn = torch.softmax(attn, dim=-1)  
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(B, -1, H, W)

        x = self.norm(x) + lepe

        with torch.cuda.amp.autocast(enabled=False):
            # disable amp for better training stability
            x = x.to(torch.float32)
            gate = gate.to(torch.float32)
            x = x * gate
            x = self.proj(x)
        
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0): 
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



class Block(nn.Module):
    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=256, 
                 drop_path=0, 
                 layerscale=False,
                 resscale=False,
                 layer_init_values=1e-6,
                 token_mixer=Attention,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d):
        super().__init__()
        
        self.layerscale = layerscale
        self.resscale = resscale

        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = token_mixer(embed_dim, num_heads, window_size, window_dilation, global_mode, image_size, use_rpb, sr_ratio)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = channel_mixer(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale or resscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc):
        if self.resscale:
            x = x + self.cpe1(x)
            x = self.layer_scale1(x) + self.drop_path(self.token_mixer(self.norm1(x), pos_enc))
            x = x + self.cpe2(x)
            x = self.layer_scale2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.cpe1(x)
            x = x + self.drop_path(self.layer_scale1(self.token_mixer(self.norm1(x), pos_enc)))
            x = x + self.cpe2(x)
            x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x
    

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 image_size=None,
                 embed_dim=64, 
                 depth=4, 
                 num_heads=4,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=96, 
                 drop_path=0,
                 layerscale=False,
                 resscale=False,
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 use_checkpoint=0,
            ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.rope = RoPE(embed_dim, num_heads)
        
        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_dim=embed_dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          window_dilation=window_dilation,
                          global_mode=global_mode,
                          ffn_dim=ffn_dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layerscale=layerscale,
                          resscale=resscale,
                          layer_init_values=layer_init_values,
                          norm_layer=norm_layer,
                          image_size=image_size,
                          use_rpb=use_rpb,
                          sr_ratio=sr_ratio,
            )
            self.blocks.append(block)

    def forward(self, x):
        pos_enc = self.rope((x.shape[2:]))
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, pos_enc, use_reentrant=False)
            else:
                x = blk(x, pos_enc)
        return x


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )


class A2Mamba(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=[7, 7, 7, 7],
                 window_dilation=[1, 1, 1, 1],
                 use_rpb=False,
                 sr_ratio=[8, 4, 2, 1],
                 mlp_ratios=[4, 4, 4, 4], 
                 drop_rate=0,
                 drop_path_rate=0,
                 projection=1024,
                 layerscales=[False, False, False, False],
                 resscales=[False, False, False, False],
                 layer_init_values=[1, 1, 1, 1],
                 norm_layer=LayerNorm2d,
                 return_features=False,
                 use_checkpoint=[0, 0, 0, 0]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.return_features = return_features

        # split image into non-overlapping patches
        self.patch_embed = stem(in_chans=in_chans, embed_dim=embed_dims[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # input resolution
        image_size = to_2tuple(image_size)
        image_size = [(image_size[0]//2**(i+2), image_size[1]//2**(i+2)) for i in range(4)]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                window_dilation=window_dilation[i_layer],
                global_mode=(i_layer==3),
                use_rpb=use_rpb,
                sr_ratio=sr_ratio[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                layerscale=layerscales[i_layer],
                resscale=resscales[i_layer],
                layer_init_values=layer_init_values[i_layer],
                norm_layer=norm_layer,
                image_size=image_size[i_layer],
                use_checkpoint=use_checkpoint[i_layer],
            )
                       
            downsample = nn.Sequential(
                nn.Conv2d(embed_dims[i_layer], embed_dims[i_layer+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims[i_layer+1])
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            
            self.layers.append(layer)
            self.layers.append(downsample)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, projection, kernel_size=1),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        out = []
        idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i%2 == 0:
                out.append(x)
                idx += 1

        return tuple(out)

    def forward(self, x):
        if self.return_features:
            return self.forward_features(x)
        else:
            x = self.patch_embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.classifier(x).flatten(1)
        return x
    
    def flops(self, shape=(3, 224, 224)):
        
        supported_ops={
            "prim::PythonOp.SelectiveScanStateFn": selective_scan_state_flop_jit,
            "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": av_2d_flop,
        }

        model = copy.deepcopy(self)
        
        if torch.cuda.is_available:
            model.cuda()
        model.eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return (sum(Gflops.values()) * 1e9, params)



def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',  # 'bilinear' or 'bicubic'
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }



@register_model
def a2mamba_n(pretrained=None, pretrained_cfg=None, **args):
    # Nano model
    model = A2Mamba(
        embed_dims=[32, 64, 128, 192],
        depths=[2, 2, 8, 2],
        num_heads=[2, 2, 4, 8],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[False, False, False, False],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        **args,
    )
    model.default_cfg = _cfg()

    if pretrained:
        pretrained = 'https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_nano.pth'
        load_checkpoint(model, pretrained)

    return model



@register_model
def a2mamba_t(pretrained=None, pretrained_cfg=None, **args):
    # Tiny model
    model = A2Mamba(
        embed_dims=[48, 96, 256, 448],
        depths=[2, 2, 10, 2],
        num_heads=[2, 4, 8, 16],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[False, False, False, False],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        **args,
    )
    model.default_cfg = _cfg()

    if pretrained:
        pretrained = 'https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_tiny.pth'
        load_checkpoint(model, pretrained)

    return model


@register_model
def a2mamba_s(pretrained=None, pretrained_cfg=None, **args):
    # Small model
    model = A2Mamba(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 4, 12, 4],
        num_heads=[2, 4, 10, 16],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        resscales=[False, False, True, True],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        **args,
    )
    model.default_cfg = _cfg()

    if pretrained:
        pretrained = 'https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_small.pth'
        load_checkpoint(model, pretrained)

    return model


@register_model
def a2mamba_b(pretrained=None, pretrained_cfg=None, **args):
    # Base model
    model = A2Mamba(
        embed_dims=[96, 192, 384, 512],
        depths=[4, 6, 12, 6],
        num_heads=[4, 8, 12, 16],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3.75, 3.75],
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1e-6, 1e-6],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        **args,
    )
    model.default_cfg = _cfg(crop_pct=0.95)

    if pretrained:
        pretrained = 'https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_base.pth'
        load_checkpoint(model, pretrained)

    return model


@register_model
def a2mamba_l(pretrained=None, pretrained_cfg=None, **args):
    # Large model
    model = A2Mamba(
        embed_dims=[96, 192, 448, 672],
        depths=[4, 6, 18, 6],
        num_heads=[4, 8, 16, 24],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 4, 4],
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1e-6, 1e-6],
        use_rpb=True,
        norm_layer=LayerNorm2d,
        **args,
    )
    model.default_cfg = _cfg(crop_pct=0.975)

    if pretrained:
        pretrained = 'https://huggingface.co/LMMM2025/A2Mamba/resolve/main/a2mamba_pretrained_weights/a2mamba_large.pth'
        load_checkpoint(model, pretrained)

    return model