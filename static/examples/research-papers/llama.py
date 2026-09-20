"""A complete narrow LLaMA-style decoder trained on a local character corpus.
Includes RMSNorm, rotary Q/K positions, causal attention and SwiGLU.
Teaching adaptation: no SentencePiece, KV cache or distributed training.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight

def rotary(x):
    # [batch, heads, length, head_dim]; adjacent dimension pairs share a rotation.
    d=x.size(-1)
    frequency=10000**(-torch.arange(0,d,2,device=x.device).float()/d)
    angle=torch.arange(x.size(-2),device=x.device)[:,None]*frequency
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),-1).flatten(-2)

class Layer(nn.Module):
    def __init__(self,d=32,heads=4):
        super().__init__(); self.heads=heads; self.d=d
        self.n1,self.n2=RMSNorm(d),RMSNorm(d)
        self.q,self.k,self.v,self.out=[nn.Linear(d,d,bias=False) for _ in range(4)]
        # Approximately 8d/3 hidden units keeps SwiGLU parameters near a 4d FFN.
        hidden=88
        self.gate,self.up,self.down=nn.Linear(d,hidden,bias=False),nn.Linear(d,hidden,bias=False),nn.Linear(hidden,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape; h=self.n1(x)
        def split(z): return z.reshape(b,t,self.heads,d//self.heads).transpose(1,2)
        q,k,v=rotary(split(self.q(h))),rotary(split(self.k(h))),split(self.v(h))
        scores=q@k.transpose(-2,-1)/math.sqrt(d//self.heads)
        scores=scores.masked_fill(torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1),float('-inf'))
        context=(scores.softmax(-1)@v).transpose(1,2).reshape(b,t,d)
        x=x+self.out(context)
        h=self.n2(x)
        return x+self.down(F.silu(self.gate(h))*self.up(h))

class LLaMA(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embed=nn.Embedding(vocab,32)
        self.layers=nn.ModuleList([Layer() for _ in range(2)])
        self.norm,self.output=RMSNorm(32),nn.Linear(32,vocab,bias=False)
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.layers: x=layer(x)
        return self.output(self.norm(x))

text=('small models can learn patterns. more data gives more practice.\n')*50
alphabet=sorted(set(text)); vocab={c:i for i,c in enumerate(alphabet)}
data=torch.tensor([vocab[c] for c in text])
model=LLaMA(len(vocab)); optim=torch.optim.AdamW(model.parameters(),lr=.003)
for step in range(250):
    starts=torch.randint(len(data)-33,(16,))
    rows=torch.stack([data[s:s+33] for s in starts])
    logits=model(rows[:,:-1])
    loss=F.cross_entropy(logits.reshape(-1,len(vocab)),rows[:,1:].reshape(-1))
    optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
model.eval(); prefix=torch.tensor([[vocab[c] for c in 'small ']])
with torch.no_grad():
    for _ in range(50):
        logits=model(prefix[:,-32:])[:,-1]
        prefix=torch.cat((prefix,logits.argmax(-1,keepdim=True)),-1)
print(''.join(alphabet[i] for i in prefix[0]));print('Loss:',loss.item())
assert torch.isfinite(loss)
torch.save(model.state_dict(),'llama-demo.pt')
