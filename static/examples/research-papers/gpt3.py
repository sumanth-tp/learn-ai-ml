"""In-context learning on changing symbol dictionaries using a trained causal LM.
The small synthetic experiment is not GPT-3 or a natural-language benchmark.
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

# IDs 0..3 are keys; 4..7 are values; 8 separates demonstrations from a query.
# Each row has a new random mapping, so a fixed key-to-value answer cannot work.
def batch(n):
    rows = []
    for _ in range(n):
        mapping = torch.randperm(4) + 4
        order = torch.randperm(4)
        query = torch.randint(4, ()).item()
        pairs = [item for key in order.tolist() for item in (key, mapping[key].item())]
        rows.append(pairs + [8, query, mapping[query].item()])
    return torch.tensor(rows)

model = LanguageModel(9, context=12)
optim = torch.optim.AdamW(model.parameters(), lr=.003)
for step in range(700):
    rows = batch(32)
    # All tokens use the same next-token objective. Random values are irreducible.
    loss = lm_loss(model, rows)
    optim.zero_grad(); loss.backward(); optim.step()
model.eval()
heldout = batch(256)
weights_before = {k: v.clone() for k, v in model.state_dict().items()}
with torch.no_grad():
    answer = model(heldout[:, :-1])[:, -1].argmax(-1)
    accuracy = (answer == heldout[:, -1]).float().mean().item()
    example = heldout[:1]
    print('Four-shot prompt:', example[0, :-1].tolist())
    print('Expected:', example[0, -1].item(), 'predicted:', answer[0].item())
    for shots in (0, 1, 4):
        # Keep only the chosen number of demonstrations, then separator + query.
        prefix = torch.cat((example[:, :2*shots], example[:, -3:-1]), dim=1)
        print(shots, 'shot value probabilities:', model(prefix)[:, -1].softmax(-1)[0, 4:8].tolist())
assert all(torch.equal(weights_before[k], v) for k, v in model.state_dict().items())
print('Held-out mapping accuracy:', round(accuracy, 3), '(chance = 0.25)')
torch.save(model.state_dict(), 'gpt3-demo.pt')
