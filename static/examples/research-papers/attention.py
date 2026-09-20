"""Train a small encoder-decoder Transformer to reverse token sequences.
Implements the original post-norm structure, sinusoidal positions and teacher forcing.
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


class Positions(nn.Module):
    def __init__(self, width=32, length=32):
        super().__init__()
        pos = torch.arange(length).float()[:, None]
        freq = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000)/width))
        pe = torch.zeros(length, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention(32, 4)
        self.ff = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
        self.n1, self.n2 = nn.LayerNorm(32), nn.LayerNorm(32)
    def forward(self, x):
        x = self.n1(x + self.attn(x))
        return self.n2(x + self.ff(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(32, 4), Attention(32, 4)
        self.ff = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
        self.norm = nn.ModuleList([nn.LayerNorm(32) for _ in range(3)])
    def forward(self, x, memory):
        x = self.norm[0](x + self.self_attn(x, causal=True))
        x = self.norm[1](x + self.cross_attn(x, memory))
        return self.norm[2](x + self.ff(x))

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding, self.positions = nn.Embedding(12, 32), Positions()
        self.encoders = nn.ModuleList([Encoder() for _ in range(2)])
        self.decoders = nn.ModuleList([Decoder() for _ in range(2)])
        self.output = nn.Linear(32, 12)
    def encode(self, source):
        x = self.positions(self.embedding(source)*math.sqrt(32))
        for layer in self.encoders: x = layer(x)
        return x
    def decode(self, target, memory):
        x = self.positions(self.embedding(target)*math.sqrt(32))
        for layer in self.decoders: x = layer(x, memory)
        return self.output(x)
    def forward(self, source, target):
        return self.decode(target, self.encode(source))

def batch(n):
    source = torch.randint(3, 12, (n, 4))
    # ID 1 starts decoding; ID 2 ends it. Fixed lengths require no padding mask.
    target = torch.cat((torch.ones(n, 1, dtype=torch.long), source.flip(1),
                        torch.full((n, 1), 2)), dim=1)
    return source, target

model = Transformer()
optim = torch.optim.Adam(model.parameters(), lr=.003)
for step in range(600):
    source, target = batch(32)
    logits = model(source, target[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 12), target[:, 1:].reshape(-1), label_smoothing=.1)
    optim.zero_grad(); loss.backward(); optim.step()
model.eval()
source, target = batch(64)
with torch.no_grad():
    memory = model.encode(source)
    generated = torch.ones(64, 1, dtype=torch.long)
    for _ in range(5):
        next_id = model.decode(generated, memory)[:, -1].argmax(-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
print('Source:', source[0].tolist(), 'generated:', generated[0].tolist())
print('Held-out exact sequence accuracy:', (generated == target).all(1).float().mean().item())
assert generated.shape == target.shape
torch.save(model.state_dict(), 'transformer-demo.pt')
