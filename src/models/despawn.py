# AudioAnomalyDetector/src/models/despawn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Kernel(nn.Module):
    def __init__(self, kernel_init=8, train_kern=True):
        super().__init__()
        if isinstance(kernel_init, int):
            self.kernel = nn.Parameter(torch.randn(kernel_init,1,1,1), requires_grad=train_kern)
        else:
            arr = torch.tensor(kernel_init,dtype=torch.float32)
            self.kernel = nn.Parameter(arr.reshape(-1,1,1,1), requires_grad=train_kern)
    def forward(self,x):
        return self.kernel

class LowPassWave(nn.Module):
    def forward(self, x, kernel):
        x = x.permute(0,2,1,3)
        k = kernel.permute(3,2,0,1)
        pad = (k.shape[2]-1)//2
        y = F.conv2d(F.pad(x,(0,0,pad,pad),mode='reflect'),k,stride=(2,1))
        return y.permute(0,2,1,3)

class HighPassWave(nn.Module):
    def forward(self, x, kernel):
        q = torch.tensor([(-1)**i for i in range(kernel.shape[0])],device=x.device)
        flip = torch.flip(kernel,[0])*q.view(-1,1,1,1)
        x = x.permute(0,2,1,3)
        k = flip.permute(3,2,0,1)
        pad = (k.shape[2]-1)//2
        y = F.conv2d(F.pad(x,(0,0,pad,pad),mode='reflect'),k,stride=(2,1))
        return y.permute(0,2,1,3)

class LowPassTrans(nn.Module):
    def forward(self, x, kernel, output_size):
        x = x.permute(0,2,1,3)
        k = kernel.permute(3,2,0,1)
        pad = (k.shape[2]-1)//2
        y = F.conv_transpose2d(F.pad(x,(0,0,pad,pad),mode='reflect'),k,stride=(2,1),padding=pad,
                               output_padding=(output_size[1] - ((x.shape[2]-1)*2 + k.shape[2] - 2*pad),0))
        return y.permute(0,2,1,3)

class HighPassTrans(nn.Module):
    def forward(self, x, kernel, output_size):
        q = torch.tensor([(-1)**i for i in range(kernel.shape[0])],device=x.device)
        flip = torch.flip(kernel,[0])*q.view(-1,1,1,1)
        x = x.permute(0,2,1,3)
        k = flip.permute(3,2,0,1)
        pad = (k.shape[2]-1)//2
        y = F.conv_transpose2d(F.pad(x,(0,0,pad,pad),mode='reflect'),k,stride=(2,1),padding=pad,
                               output_padding=(output_size[1] - ((x.shape[2]-1)*2 + k.shape[2] - 2*pad),0))
        return y.permute(0,2,1,3)

class HardThresholdAssym(nn.Module):
    def __init__(self, init=1.0, train_bias=True):
        super().__init__()
        self.thr_p = nn.Parameter(torch.ones(1,1,1,1)*init, requires_grad=train_bias)
        self.thr_n = nn.Parameter(torch.ones(1,1,1,1)*init, requires_grad=train_bias)
    def forward(self,x):
        p,n = self.thr_p.to(x.device), self.thr_n.to(x.device)
        a = 10
        return x*(torch.sigmoid(a*(x-p))+torch.sigmoid(-a*(x+n)))

class ChannelAttention(nn.Module):
    def __init__(self, reduction_ratio=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(1,1//reduction_ratio or 1,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(1//reduction_ratio or 1,1,1,bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x = x.permute(0,2,1,3)
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        s = self.sig(a+m)
        return s.permute(0,2,1,3)

class EnhancedDeSpaWN(nn.Module):
    def __init__(self, kernel_init, kern_trainable, level, kernels_constraint, init_ht, train_ht):
        super().__init__()
        self.level = level
        self.low_wave = LowPassWave()
        self.high_wave = HighPassWave()
        self.low_trans = LowPassTrans()
        self.high_trans = HighPassTrans()
        self.kernels = nn.ModuleList([Kernel(kernel_init,kern_trainable) for _ in range(level)])
        self.hts = nn.ModuleList([HardThresholdAssym(init_ht,train_ht) for _ in range(level)])
        self.final_ht = HardThresholdAssym(init_ht,train_ht)
        self.attn = ChannelAttention()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,1,3,padding=1), nn.BatchNorm2d(1)
        )
    def forward(self,x,return_coeffs=False):
        g = self.attn(x)*x
        coeffs=[]
        shapes=[]
        for i in range(self.level):
            shapes.append(g.shape)
            h = self.high_wave(g,self.kernels[i].kernel)
            h = self.hts[i](h); coeffs.append(h)
            g = self.low_wave(g,self.kernels[i].kernel)
        g = self.final_ht(g)
        for i in reversed(range(len(coeffs))):
            h = self.high_trans(coeffs[i],self.kernels[i].kernel, shapes[i])
            g = self.low_trans(g,self.kernels[i].kernel, shapes[i]) + h
        p = self.conv(g.permute(0,2,1,3)).permute(0,2,1,3)
        if return_coeffs:
            return p, coeffs
        return p
