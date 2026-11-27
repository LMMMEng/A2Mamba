'''
This is a plug-and-play implementation of MASS (Multi-scale Attention-augmented
State Space) block in the paper:
https://arxiv.org/abs/2507.16624
'''
import math
import warnings
import torch
from torch import nn
from einops import rearrange, repeat
from timm.models.pvt_v2 import to_2tuple
from natten.functional import na2d_av, na2d_qk

try:
    import selective_scan_cuda_oflex_rh
    has_selective_scan = True
except:
    has_selective_scan = False
    warnings.warn("The MASS block requires the 'selective_scan_cuda_oflex_rh' CUDA extension to run."
                  " Please build it from the 'selective_scan' folder for full functionality.")


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


class MASS(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 num_heads=2,
                 window_size=7,
                 window_dilation=1,
                 use_rpb=False):

        super().__init__()
        '''
        Multi-scale Attention-augmented State Space (MASS) block, the token mixer of
        A2Mamba. It takes and returns a tensor of shape (B, C, H, W).
        Args:
            embed_dim (int): The dimension of the input and output features.
            num_heads (int): Number of attention heads. The head dimension is
                embed_dim // num_heads, and the heads are evenly split into two
                groups for the two attention branches described below.
            window_size (int | tuple(int, int)): Neighborhood window size (kh, kw)
                shared by the two attention branches.
            window_dilation (int | tuple(int, int)): Neighborhood dilation
                (dh, dw) of the first attention branch. The second branch uses
                the adaptive dilation (min(H // kh, kh), min(W // kw, kw)),
                which yields the multi-scale (reciprocal) attention.
            use_rpb (bool): If True, use relative position bias in the two
                attention branches.
        '''
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation

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

        self.ssm = A2SSM(d_model=embed_dim, num_heads=num_heads)
        self.rope = RoPE(embed_dim, num_heads)

        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads//2, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)] * 2
            self.rpb = nn.ParameterList(rpb_list)

        self.reset_parameters()


    def reset_parameters(self):
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item) # which better? nn.init.trunc_normal_(item, std=0.02)


    def forward(self, x):

        B, C, H, W = x.shape
        dilation = [int(H/self.window_size[0]), int(W/self.window_size[1])]
        dilation = [min(dilation[0], self.window_size[0]), min(dilation[1], self.window_size[1])]

        gate = self.gate(x)
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])

        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)

        sin, cos = self.rope((H, W))
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

        state_params = [attn1, attn2, self.window_size, dilation]
        x = self.ssm(x, state_params=state_params)

        x = self.norm(x) + lepe

        with torch.cuda.amp.autocast(enabled=False):
            # disable amp for better training stability
            x = x.to(torch.float32)
            gate = gate.to(torch.float32)
            x = x * gate
            x = self.proj(x)

        return x


if __name__ == '__main__':

    from timm.utils import random_seed
    random_seed(6)

    device = torch.device('cuda')
    x = torch.randn(1, 64, 32, 32).to(device)
    model = MASS(embed_dim=64,
                 num_heads=2,
                 window_size=[11, 9],
                 window_dilation=1,
                 use_rpb=True).to(device)
    print(model)
    model.eval()

    y = model(x)
    print(y.shape)
