"""Run cold-start SFT, token-level GRPO, rejection sampling and distillation.
Teaching adaptation: tiny arithmetic answer sequences, synthetic supervision.
No claim of reproducing DeepSeek's reasoning or its full multi-stage dataset mix.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7);torch.set_num_threads(1)
# A prompt is an integer 0..3; response is [answer, EOS], answer=(prompt+1)%4.
# Tokens 0..3 are answers; EOS=4, BOS=5.
class Policy(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.prompt=nn.Embedding(4,width);self.token=nn.Embedding(6,width)
        self.rnn=nn.GRU(width,width,batch_first=True);self.head=nn.Linear(width,5)
    def forward(self,q,prefix):
        h,_=self.rnn(self.token(prefix),self.prompt(q)[None])
        return self.head(h)
    @torch.no_grad()
    def sample(self,q):
        prefix=torch.full((len(q),1),5)
        for _ in range(2):
            token=torch.distributions.Categorical(logits=self(q,prefix)[:,-1]).sample()
            prefix=torch.cat((prefix,token[:,None]),1)
        return prefix[:,1:]

def token_logp(model,q,response):
    prefix=torch.cat((torch.full((len(q),1),5),response[:,:-1]),1)
    return model(q,prefix).log_softmax(-1).gather(-1,response[:,:,None]).squeeze(-1)

def targets(q): return torch.stack(((q+1)%4,torch.full_like(q,4)),1)
def reward(q,response):
    correct=(response[:,0]==(q+1)%4).float()
    formatted=(response[:,1]==4).float()
    return correct+0.25*formatted

def sft(model,q,y,steps,lr=.02):
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps):
        loss=-token_logp(model,q,y).mean()
        optim.zero_grad();loss.backward();optim.step()

policy=Policy();q=torch.arange(4)
sft(policy,q,targets(q),steps=3)  # Cold start; R1-Zero would skip this stage.
reference=copy.deepcopy(policy).eval()
for p in reference.parameters():p.requires_grad_(False)
optim=torch.optim.Adam(policy.parameters(),lr=.005)
group=16
for rollout in range(80):
    prompts=torch.arange(4).repeat_interleave(group)
    with torch.no_grad():
        response=policy.sample(prompts)
        rewards=reward(prompts,response).reshape(4,group)
        advantage=(rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,correction=0)+1e-8)
        advantage=advantage.reshape(-1,1)
        old_logp=token_logp(policy,prompts,response)
        ref_logp=token_logp(reference,prompts,response)
        # Include the first EOS token but exclude tokens after it.
        mask=torch.ones_like(response,dtype=torch.float)
        mask[:,1]=(response[:,0]!=4).float()
    for _ in range(2):
        logp=token_logp(policy,prompts,response)
        ratio=(logp-old_logp).exp()
        clipped=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
        log_ratio=ref_logp-logp
        kl=log_ratio.exp()-log_ratio-1
        per_token=clipped-.02*kl
        objective=((per_token*mask).sum(1)/mask.sum(1)).mean()
        optim.zero_grad();(-objective).backward()
        nn.utils.clip_grad_norm_(policy.parameters(),1.);optim.step()
# Rejection sampling collects verified answers, then supervised distillation.
with torch.no_grad():
    prompts=q.repeat_interleave(128);responses=policy.sample(prompts)
    accepted=(responses==targets(prompts)).all(1)
    print('Verified response fraction:',accepted.float().mean().item())
    train_q,train_y=prompts[accepted],responses[accepted]
assert accepted.any()
student=Policy(width=12)
sft(student,train_q,train_y,steps=100)
with torch.no_grad():
    student_response=student.sample(q.repeat_interleave(64))
    accuracy=(student_response==targets(q.repeat_interleave(64))).all(1).float().mean()
print('Distilled student exact-answer rate:',accuracy.item())
torch.save({'teacher':policy.state_dict(),'student':student.state_dict()},'r1-demo.pt')
