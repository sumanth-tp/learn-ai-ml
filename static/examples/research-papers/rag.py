"""Train a tiny dense retriever and conditional generator with both RAG losses.
Teaching adaptation: exhaustive search over four documents, small Transformer.
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

# The corpus is public toy data; IDs represent whole words.
words = '<bos> <eos> france germany italy spain capital paris berlin rome madrid'.split()
vocab = {w: i for i, w in enumerate(words)}
def encode(text): return [vocab[w] for w in text.split()]
documents = torch.tensor([encode(s) for s in ['france capital paris', 'germany capital berlin',
                                             'italy capital rome', 'spain capital madrid']])
queries = torch.tensor([encode(s) for s in ['france capital', 'germany capital', 'italy capital', 'spain capital']])
answers = torch.tensor([[7, 1], [8, 1], [9, 1], [10, 1]])

class RAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Embedding(len(words), 32)
        self.document = nn.Embedding(len(words), 32)
        self.document.weight.requires_grad_(False)
        self.generator = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=64, dropout=0., batch_first=True)
        self.tokens, self.positions = nn.Embedding(len(words), 32), nn.Embedding(16, 32)
        self.output = nn.Linear(32, len(words))
        # Build once: the original system likewise keeps its document index fixed.
        self.register_buffer('index', self.document(documents).mean(1).detach())
    def retrieve(self, query, k):
        scores = self.query(query).mean(1) @ self.index.T
        scores, doc_ids = scores.topk(k, dim=-1)
        return scores.log_softmax(-1), doc_ids
    def token_log_probs(self, query, doc_ids, prefix):
        b, k = doc_ids.shape
        source = torch.cat((query[:, None].expand(-1,k,-1), documents[doc_ids]), -1).reshape(b*k,-1)
        target = prefix[:, None].expand(-1,k,-1).reshape(b*k,-1)
        src = self.tokens(source) + self.positions(torch.arange(source.size(1)))
        tgt = self.tokens(target) + self.positions(torch.arange(target.size(1)))
        mask = torch.ones(target.size(1), target.size(1), dtype=torch.bool).triu(1)
        logits = self.output(self.generator(src, tgt, tgt_mask=mask))
        return logits.log_softmax(-1).reshape(b,k,target.size(1),-1)
    def loss(self, query, answer, mode):
        log_docs, doc_ids = self.retrieve(query, k=4)
        prefix = torch.cat((torch.zeros(len(query),1,dtype=torch.long), answer[:,:-1]),-1)
        log_tokens = self.token_log_probs(query, doc_ids, prefix)
        chosen = log_tokens.gather(-1, answer[:,None,:,None].expand(-1,4,-1,1)).squeeze(-1)
        if mode == 'sequence':
            log_likelihood = torch.logsumexp(log_docs + chosen.sum(-1), dim=1)
        else:
            log_likelihood = torch.logsumexp(log_docs[:,:,None] + chosen, dim=1).sum(-1)
        return -log_likelihood.mean()
    @torch.no_grad()
    def generate(self, query, mode):
        log_docs, doc_ids = self.retrieve(query, 4)
        prefix = torch.zeros(len(query),1,dtype=torch.long)
        posterior = log_docs.clone()
        for _ in range(2):
            conditional = self.token_log_probs(query, doc_ids, prefix)[:,:,-1]
            mixed = torch.logsumexp(posterior[:,:,None] + conditional, dim=1)
            token = mixed.argmax(-1)
            if mode == 'sequence':
                # A shared latent document: update p(document | generated prefix).
                evidence = conditional.gather(-1, token[:,None,None].expand(-1,4,1)).squeeze(-1)
                posterior = (posterior + evidence).log_softmax(-1)
            prefix = torch.cat((prefix, token[:,None]), -1)
        return prefix[:,1:]

for mode in ('sequence', 'token'):
    model = RAG()
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.005)
    for step in range(200):
        loss = model.loss(queries, answers, mode)
        optim.zero_grad(); loss.backward(); optim.step()
    model.eval()
    result = model.generate(queries, mode)
    print(mode, 'loss:', round(loss.item(), 3))
    print([' '.join(words[i] for i in row) for row in result])
    assert model.document.weight.grad is None
    torch.save(model.state_dict(), 'rag-' + mode + '-demo.pt')
