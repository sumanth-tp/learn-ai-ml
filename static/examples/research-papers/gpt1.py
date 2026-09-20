"""Small GPT-style pre-training followed by supervised task fine-tuning.
Teaching adaptation: word tokens, two narrow layers and a synthetic corpus.
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

class LanguageModel(nn.Module):
    def __init__(self, vocab, context=64, width=32, pre_norm=True):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, pre_norm=pre_norm) for _ in range(2)])
        self.norm = nn.LayerNorm(width) if pre_norm else nn.Identity()
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.token.weight

    def hidden(self, ids):
        assert ids.size(1) <= self.context
        x = self.token(ids) + self.position(torch.arange(ids.size(1), device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.hidden(ids))

    @torch.no_grad()
    def generate(self, ids, count, temperature=1., top_k=None):
        self.eval()
        for _ in range(count):
            logits = self(ids[:, -self.context:])[:, -1] / temperature
            if top_k is not None:
                cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            token = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat((ids, token), dim=1)
        return ids

def lm_loss(model, rows):
    logits = model(rows[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), rows[:, 1:].reshape(-1))

words = '<pad> <start> good bad film story happy sad <end>'.split()
vocab = {word: i for i, word in enumerate(words)}
texts = ['<start> good film happy <end>', '<start> good story happy <end>',
         '<start> bad film sad <end>', '<start> bad story sad <end>']
rows = torch.tensor([[vocab[w] for w in text.split()] for text in texts])
labels = torch.tensor([1, 1, 0, 0])
model = LanguageModel(len(words), pre_norm=False)
optim = torch.optim.AdamW(model.parameters(), lr=.003)
for step in range(160):
    loss = lm_loss(model, rows)
    optim.zero_grad(); loss.backward(); optim.step()
print('Pre-training loss:', round(loss.item(), 3))
# A new classification head is trained together with the pre-trained backbone.
classifier = nn.Linear(32, 2)
optim = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=.001)
for step in range(100):
    logits = classifier(model.hidden(rows)[:, -1])
    task = F.cross_entropy(logits, labels)
    loss = task + .5 * lm_loss(model, rows)
    optim.zero_grad(); loss.backward(); optim.step()
with torch.no_grad():
    predictions = classifier(model.hidden(rows)[:, -1]).argmax(-1)
print('Sentiment predictions:', predictions.tolist(), 'targets:', labels.tolist())
assert torch.equal(predictions, labels)
torch.save({'backbone': model.state_dict(), 'classifier': classifier.state_dict(),
            'vocab': vocab}, 'gpt1-demo.pt')
