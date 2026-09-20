"""Train MLM + NSP, then fine-tune a small bidirectional encoder.
Teaching adaptation: integer tokens and synthetic sentence pairs, no WordPiece.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)

class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads, self.size = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query, memory=None, causal=False):
        memory = query if memory is None else memory
        b, t, d = query.shape
        def split(x):
            return x.reshape(b, -1, self.heads, self.size).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(memory)), split(self.v(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.size)
        if causal:
            mask = torch.ones(t, k.size(-2), dtype=torch.bool, device=query.device).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))
        context = scores.softmax(-1) @ v
        return self.out(context.transpose(1, 2).reshape(b, t, d))

class Block(nn.Module):
    def __init__(self, width=32, heads=4, pre_norm=True):
        super().__init__()
        self.attn = Attention(width, heads)
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.pre_norm = pre_norm

    def forward(self, x, causal=True):
        if self.pre_norm:
            x = x + self.attn(self.n1(x), causal=causal)
            return x + self.ff(self.n2(x))
        x = self.n1(x + self.attn(x, causal=causal))
        return self.n2(x + self.ff(x))


# Specials: PAD=0, CLS=1, SEP=2, MASK=3. Two topics occupy disjoint token groups.
def batch(n):
    topic = torch.randint(2, (n,))
    is_next = torch.randint(2, (n,))
    other_topic = torch.where(is_next.bool(), topic, 1-topic)
    first = torch.randint(4, (n, 3)) + 4 + topic[:, None]*4
    second = torch.randint(4, (n, 3)) + 4 + other_topic[:, None]*4
    ids = torch.cat((torch.ones(n, 1, dtype=torch.long), first, torch.full((n, 1), 2),
                     second, torch.full((n, 1), 2)), dim=1)
    return ids, 1-is_next, topic  # NSP label 0 means IsNext.

def corrupt(ids):
    selected = (torch.rand(ids.shape) < .15) & (ids >= 4)
    # Ensure this tiny batch always has a supervised token.
    if not selected.any(): selected[0, 1] = True
    labels = ids.clone().masked_fill(~selected, -100)
    draw = torch.rand(ids.shape)
    inputs = ids.clone()
    inputs[selected & (draw < .8)] = 3
    random_tokens = torch.randint(4, 12, ids.shape)
    random_mask = selected & (draw >= .8) & (draw < .9)
    inputs[random_mask] = random_tokens[random_mask]
    return inputs, labels

class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.word, self.pos, self.segment = nn.Embedding(12, 32), nn.Embedding(9, 32), nn.Embedding(2, 32)
        self.input_norm = nn.LayerNorm(32)
        self.blocks = nn.ModuleList([Block(pre_norm=False) for _ in range(2)])
        self.mlm_transform = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.LayerNorm(32))
        self.mlm = nn.Linear(32, 12)
        self.mlm.weight = self.word.weight
        self.pool = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.nsp = nn.Linear(32, 2)
    def hidden(self, ids):
        segments = torch.tensor([0,0,0,0,0,1,1,1,1])
        x = self.input_norm(self.word(ids) + self.pos(torch.arange(9)) + self.segment(segments))
        for layer in self.blocks: x = layer(x, causal=False)
        return x
    def forward(self, ids):
        x = self.hidden(ids)
        return self.mlm(self.mlm_transform(x)), self.nsp(self.pool(x[:, 0]))

model = Bert()
optim = torch.optim.AdamW(model.parameters(), lr=.003)
for step in range(300):
    ids, next_labels, _ = batch(32)
    inputs, targets = corrupt(ids)
    mlm, nsp = model(inputs)
    loss = F.cross_entropy(mlm.reshape(-1, 12), targets.reshape(-1)) + F.cross_entropy(nsp, next_labels)
    optim.zero_grad(); loss.backward(); optim.step()
print('Pre-training loss:', round(loss.item(), 3))
# Fine-tune the whole encoder for topic classification of the first sentence.
head = nn.Linear(32, 2)
optim = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.001)
for step in range(100):
    ids, _, topic = batch(32)
    loss = F.cross_entropy(head(model.hidden(ids)[:, 0]), topic)
    optim.zero_grad(); loss.backward(); optim.step()
with torch.no_grad():
    ids, _, topic = batch(256)
    accuracy = (head(model.hidden(ids)[:, 0]).argmax(-1) == topic).float().mean()
print('Held-out topic accuracy:', accuracy.item())
assert torch.isfinite(loss)
torch.save({'encoder': model.state_dict(), 'head': head.state_dict()}, 'bert-demo.pt')
