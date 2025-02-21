import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pscan import pscan

@dataclass
class MambaConfig:
    d_model: int 
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 
    expand_factor: int = 2 
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True 

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model 
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i]) # type: ignore
        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False) # type: ignore
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True) # type: ignore
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale # type: ignore
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=-1) 
        x = x.transpose(1, 2) 
        x = self.conv1d(x)[:, :, :L] 
        x = x.transpose(1, 2) 
        x = F.silu(x)
        y = self.ssm(x)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output) 
        return output
    
    def ssm(self, x):
        A = -torch.exp(self.A_log.float()) 
        D = self.D.float()
        deltaBC = self.x_proj(x) 
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N) # type: ignore
        delta = F.softplus(self.dt_proj(delta)) 
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 
        BX = deltaB * (x.unsqueeze(-1)) 
        hs = pscan(deltaA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) 
        y = y + D * x
        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        
        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 
        BX = deltaB * (x.unsqueeze(-1)) 
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1) 
        y = (hs @ C.unsqueeze(-1)).squeeze(3) 
        y = y + D * x
        return y
    
    def step(self, x, cache):
        h, inputs = cache
        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=1) 
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] 
        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output) 
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) 
        cache = (h, inputs)
        return output, cache

    def ssm_step(self, x, h):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        deltaBC = self.x_proj(x) 
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)  # type: ignore
        delta = F.softplus(self.dt_proj(delta)) 
        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) 
        BX = deltaB * (x.unsqueeze(-1)) 
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) 
        h = deltaA * h + BX 
        y = (h @ C.unsqueeze(-1)).squeeze(2) 
        y = y + D * x
        return y, h.squeeze(1)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

class ResBlock1D(torch.nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(ResBlock1D,self).__init__()
        self.pre = torch.nn.Identity() if in_channel == out_channel else torch.nn.Conv1d(in_channel,out_channel,1,bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
        )
        self.relu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.pre(x)
        out = self.conv(x)
        return self.relu(x+out)


class ResConv(torch.nn.Module):
    def __init__(self,in_c,p_dropout):
        super(ResConv,self).__init__()
        self.in_c = in_c
        self.conv = torch.nn.Sequential(

            ResBlock1D(in_c,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,64),
            torch.nn.MaxPool1d(2,2),
            
            ResBlock1D(64,128),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(128,128),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),

        )
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.Linear(512,230),
        )

    def forward(self,x,idx=None):
        x = x.view(x.shape[0],self.in_c,-1)
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.cls(x)
        return x 

class XRDMamba(torch.nn.Module):
    def __init__(self,embeding_size=8,p_dropout=0.15,embed_nums=8500,m_layers=4,m_d_state=16,m_expand_factor=2,m_d_conv=3) -> None:
        super(XRDMamba,self).__init__()
        self.embed = torch.nn.Embedding(embed_nums,embeding_size)
        self.conv = ResConv(embeding_size,p_dropout)
        config = MambaConfig(d_model=embeding_size,n_layers=m_layers,d_state=m_d_state,expand_factor=m_expand_factor,d_conv=m_d_conv)
        self.mamba_embed = Mamba(config)
        
    def forward(self,x,idx):
        idx = idx.type(torch.long)
        x = x.view(x.shape[0],-1,1)
        embedding = self.embed(idx)
        features = embedding*x
        features = self.mamba_embed(features)
        features = features.transpose(1,2)
        cls = self.conv(features)
        return cls

