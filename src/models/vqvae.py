# AudioAnomalyDetector/src/models/vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .despawn import EnhancedDeSpaWN
from .transformer import EnhancedMaskedAudioTransformer

class VectorQuantizer(nn.Module):
    def __init__(self,num_embeddings,dim,commit_cost=0.25):
        super().__init__(); self.dim=dim; self.num_embeddings=num_embeddings; self.commit=commit_cost
        self.emb=nn.Embedding(num_embeddings,dim)
        self.emb.weight.data.uniform_(-1/num_embeddings,1/num_embeddings)
    def forward(self,inp):
        flat=inp.view(-1,self.dim)
        d=flat.pow(2).sum(1,keepdim=True)+self.emb.weight.pow(2).sum(1)-2*flat@self.emb.weight.t()
        idx=d.argmin(1); enc=F.one_hot(idx,self.num_embeddings).float()
        quant=enc@self.emb.weight; quant=quant.view_as(inp)
        loss=F.mse_loss(quant.detach(),inp)+self.commit*F.mse_loss(quant,inp.detach())
        return inp+(quant-inp).detach(),loss,idx

class HierarchicalFusion(nn.Module):
    def __init__(self,d_dim,t_dim,f_dim):
        super().__init__(); self.dp=nn.Linear(d_dim,f_dim); self.tp=nn.Linear(t_dim,f_dim)
        self.att=nn.MultiheadAttention(f_dim,4,batch_first=True)
        self.n1=nn.LayerNorm(f_dim);self.n2=nn.LayerNorm(f_dim)
    def forward(self,df,tf):
        d=self.dp(df).unsqueeze(1); t=self.tp(tf).unsqueeze(1)
        d2=d+self.att(self.n1(d),self.n1(t),self.n1(t))[0]
        t2=t+self.att(self.n2(t),self.n2(d),self.n2(d))[0]
        return torch.cat([d2,t2],dim=1).flatten(1)

class AudioAnomalyVQVAE(nn.Module):
    def __init__(self,despawn_model:EnhancedDeSpaWN,trans_model:EnhancedMaskedAudioTransformer,
                 fusion_dim=256,latent_dim=32,num_embeddings=512,commit=0.25):
        super().__init__()
        self.despawn=despawn_model; self.trans=trans_model
        self.f_dim=fusion_dim; self.l_dim=latent_dim
        self.fuse=HierarchicalFusion(despawn_model.level*4,trans_model.dim,fusion_dim)
        self.enc=nn.Sequential(nn.Linear(fusion_dim*2,fusion_dim),nn.LayerNorm(fusion_dim),nn.ReLU(),
                               nn.Linear(fusion_dim,latent_dim))
        self.vq=VectorQuantizer(num_embeddings,latent_dim,commit)
        self.dec=nn.Sequential(nn.Linear(latent_dim,fusion_dim),nn.ReLU(),nn.Linear(fusion_dim,fusion_dim*2))
    def forward(self,audio,return_latent=False):
        df=self.despawn.extract_features(audio); tf=self.trans.extract_features(audio)
        fused=self.fuse(df,tf)
        z=v,self.vq(fused); rec=self.dec(z)
        if return_latent: return rec,z
        return rec