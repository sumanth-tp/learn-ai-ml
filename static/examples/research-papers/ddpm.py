"""Complete DDPM training and reverse sampling on small monochrome images.
Teaching adaptation: 8x8 bars, 50 diffusion steps and a narrow time-conditioned U-Net.
Writes samples as PGM images, which can be opened by common image viewers.
"""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7);torch.set_num_threads(1)
T=50
beta=torch.linspace(.0001,.2,T)
alpha=1-beta;alpha_bar=torch.cumprod(alpha,0)
previous=torch.cat((torch.ones(1),alpha_bar[:-1]))
posterior_variance=beta*(1-previous)/(1-alpha_bar)

def extract(values,t):return values[t][:,None,None,None]
def q_sample(clean,t,noise):
    return extract(alpha_bar,t).sqrt()*clean+(1-extract(alpha_bar,t)).sqrt()*noise

def time_embedding(t,width=32):
    frequency=torch.exp(-math.log(10000)*torch.arange(width//2)/(width//2))
    angle=t.float()[:,None]*frequency
    return torch.cat((angle.sin(),angle.cos()),1)

class TimeBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.conv1,self.conv2=nn.Conv2d(cin,cout,3,padding=1),nn.Conv2d(cout,cout,3,padding=1)
        self.norm1,self.norm2=nn.GroupNorm(4,cout),nn.GroupNorm(4,cout)
        self.time=nn.Linear(32,cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,time):
        h=F.silu(self.norm1(self.conv1(x)))+self.time(time)[:,:,None,None]
        h=self.norm2(self.conv2(F.silu(h)))
        return F.silu(h+self.skip(x))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,32))
        self.down=TimeBlock(1,16);self.middle=TimeBlock(16,32)
        self.up=TimeBlock(48,16);self.output=nn.Conv2d(16,1,1)
    def forward(self,x,t):
        time=self.time(time_embedding(t))
        skip=self.down(x,time)
        middle=self.middle(F.avg_pool2d(skip,2),time)
        up=F.interpolate(middle,size=skip.shape[-2:],mode='nearest')
        return self.output(self.up(torch.cat((up,skip),1),time))

def batch(n):
    x=-torch.ones(n,1,8,8)
    for i in range(n):
        position=torch.randint(1,6,()).item()
        if torch.rand(())<.5:x[i,0,position:position+2,:]=1
        else:x[i,0,:,position:position+2]=1
    return x

model=UNet();optim=torch.optim.Adam(model.parameters(),lr=.002)
losses=[]
for step in range(600):
    clean=batch(32);t=torch.randint(T,(32,));noise=torch.randn_like(clean)
    noisy=q_sample(clean,t,noise)
    loss=F.mse_loss(model(noisy,t),noise)
    optim.zero_grad();loss.backward();optim.step();losses.append(loss.item())
model.eval()
with torch.no_grad():
    x=torch.randn(16,1,8,8)
    for step in reversed(range(T)):
        t=torch.full((len(x),),step,dtype=torch.long)
        prediction=model(x,t)
        mean=(x-beta[step]/(1-alpha_bar[step]).sqrt()*prediction)/alpha[step].sqrt()
        x=mean+posterior_variance[step].sqrt()*torch.randn_like(x) if step>0 else mean
    assert torch.isfinite(x).all()
    images=((x.clamp(-1,1)+1)*127.5).byte()
Path('ddpm-samples').mkdir(exist_ok=True)
for i,pixels in enumerate(images[:,0]):
    Path(f'ddpm-samples/{i:02d}.pgm').write_bytes(b'P5\n8 8\n255\n'+bytes(pixels.flatten().tolist()))
print('Mean first / last 50 losses:',sum(losses[:50])/50,sum(losses[-50:])/50)
print('Wrote 16 generated samples to ddpm-samples/')
torch.save(model.state_dict(),'ddpm-demo.pt')
