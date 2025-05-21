# AudioAnomalyDetector/src/models/transformer.py
import torch
import torch.nn as nn
import numpy as np
import librosa

class TemporalConv(nn.Module):
    def __init__(self,dim): super().__init__(); self.conv=nn.Conv1d(dim,dim,3,padding=1,groups=dim); self.norm=nn.LayerNorm(dim); self.act=nn.GELU()
    def forward(self,x): r=x; t=x.transpose(1,2); t=self.conv(t).transpose(1,2); return r+self.act(self.norm(t))

class EnhancedTransformerBlock(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norm2=nn.LayerNorm(dim); self.tc=TemporalConv(dim)
        self.norm3=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,x):
        a,_=self.attn(self.norm1(x),self.norm1(x),self.norm1(x)); x=x+a
        x=x+self.tc(self.norm2(x))
        return x+self.mlp(self.norm3(x))

class EnhancedMaskedAudioTransformer(nn.Module):
    def __init__(self,n_mels=128,hop_length=512,n_fft=2048,max_len=160000,
                 patch_size=16,embed_dim=768,depth=12,heads=12,mask_ratio=0.75):
        super().__init__()
        self.n_mels=n_mels; self.hop=hop_length; self.n_fft=n_fft; self.max_len=max_len
        self.patch=patch_size; self.dim=embed_dim; self.mask_ratio=mask_ratio
        self.seq_len=(max_len//hop_length)+1; self.num_patches=self.seq_len//patch_size
        self.patch_embed=nn.Linear(patch_size*n_mels,embed_dim)
        self.pos=nn.Parameter(torch.zeros(1,self.num_patches,embed_dim))
        self.mask_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.encoder=nn.ModuleList([EnhancedTransformerBlock(embed_dim,heads) for _ in range(depth)])
        self.decoder=nn.ModuleList([EnhancedTransformerBlock(embed_dim,heads) for _ in range(depth//2)])
        self.pred=nn.Linear(embed_dim,patch_size*n_mels)
        self.feat=nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim,embed_dim),nn.GELU(),nn.LayerNorm(embed_dim))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            if isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    # compute mel, patchify, unpatchify, random_masking omitted for brevity
    def forward(self,audio,return_recon=False,return_feat=False):
        # implement encoder, decoder, reconstruction…
        return audio


