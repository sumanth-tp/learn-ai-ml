"""Complete small SFT -> preference reward model -> PPO pipeline.
Teaching adaptation: a contextual bandit with three one-token answers.
One-step episodes make return = terminal reward, so GAE is unnecessary here.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
# Prompts: greeting, arithmetic, farewell. Answers: hello, four, goodbye.
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3,16)
        self.actor, self.critic = nn.Linear(16,3), nn.Linear(16,1)
    def forward(self, prompt):
        h = self.embedding(prompt)
        return self.actor(h), self.critic(h).squeeze(-1)

class Reward(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt, self.answer = nn.Embedding(3,16), nn.Embedding(3,16)
        self.score = nn.Sequential(nn.Linear(32,16), nn.Tanh(), nn.Linear(16,1))
    def forward(self, prompt, answer):
        return self.score(torch.cat((self.prompt(prompt), self.answer(answer)), -1)).squeeze(-1)

policy = Policy()
optim = torch.optim.Adam(policy.parameters(), lr=.02)
prompts = torch.arange(3)
# Stage 1: demonstrations. Deliberately brief SFT leaves room for RL improvement.
for _ in range(8):
    loss = F.cross_entropy(policy(prompts)[0], prompts)
    optim.zero_grad(); loss.backward(); optim.step()
reference = copy.deepcopy(policy).eval()
for p in reference.parameters(): p.requires_grad_(False)
# Stage 2: fixed synthetic preferences stand in for human rankings.
reward = Reward()
optim = torch.optim.Adam(reward.parameters(), lr=.01)
for _ in range(200):
    q = torch.randint(3,(32,))
    preferred = q
    rejected = (q + torch.randint(1,3,(32,))) % 3
    loss = -F.logsigmoid(reward(q,preferred)-reward(q,rejected)).mean()
    optim.zero_grad(); loss.backward(); optim.step()
reward.eval()
for p in reward.parameters(): p.requires_grad_(False)
# Stage 3: roll out the old policy, then take clipped PPO updates on that rollout.
optim = torch.optim.Adam(policy.parameters(), lr=.003)
for rollout in range(60):
    q = torch.randint(3,(64,))
    with torch.no_grad():
        old_logits, old_values = policy(q)
        old_log_probs = old_logits.log_softmax(-1)
        actions = torch.distributions.Categorical(logits=old_logits).sample()
        old_logp = old_log_probs.gather(1,actions[:,None]).squeeze(1)
        ref_log_probs = reference(q)[0].log_softmax(-1)
        ref_logp = ref_log_probs.gather(1,actions[:,None]).squeeze(1)
        # Sampled KL cost is part of the rollout reward.
        returns = reward(q,actions) - .1*(old_logp-ref_logp)
        advantage = returns-old_values
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
    for _ in range(4):
        logits, values = policy(q)
        distribution = torch.distributions.Categorical(logits=logits)
        ratio = (distribution.log_prob(actions)-old_logp).exp()
        surrogate = torch.minimum(ratio*advantage, ratio.clamp(.8,1.2)*advantage)
        actor_loss = -surrogate.mean()
        critic_loss = F.mse_loss(values,returns)
        # Tiny proxy for PPO-ptx: retain a separate language-supervision loss.
        ptx_loss = F.cross_entropy(policy(prompts)[0],prompts)
        loss = actor_loss + .5*critic_loss + .05*ptx_loss
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(),1.)
        optim.step()
with torch.no_grad():
    probabilities = policy(prompts)[0].softmax(-1)
    print('Correct-answer probabilities:', probabilities.diag().tolist())
    print('Predictions:', probabilities.argmax(-1).tolist())
assert torch.equal(probabilities.argmax(-1),prompts)
torch.save({'policy':policy.state_dict(),'reward':reward.state_dict()},'instructgpt-demo.pt')
