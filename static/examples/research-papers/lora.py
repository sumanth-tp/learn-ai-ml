"""Fit a frozen linear model's low-rank correction, save/reload and merge it.
This uses the 2021 v1 paper's 1/r scale. It is an adaptation task on synthetic data.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)

class LoRALinear(nn.Module):
    def __init__(self, base, rank=2):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features)*.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 1/rank
    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)
    @torch.no_grad()
    def merged(self):
        layer = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        layer.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if layer.bias is not None: layer.bias.copy_(self.base.bias)
        return layer

base = nn.Linear(16, 12, bias=False)
original_weight = base.weight.detach().clone()
# A low-rank target shift lets us check exactly what rank-2 adaptation can learn.
true_delta = torch.randn(12,2) @ torch.randn(2,16) * .1
x_train, x_test = torch.randn(256,16), torch.randn(128,16)
y_train = F.linear(x_train, original_weight + true_delta)
y_test = F.linear(x_test, original_weight + true_delta)
model = LoRALinear(base)
assert torch.equal(model(x_test), base(x_test))
optim = torch.optim.Adam([model.A, model.B], lr=.03)
before = F.mse_loss(model(x_test), y_test).item()
for step in range(400):
    loss = F.mse_loss(model(x_train), y_train)
    optim.zero_grad(); loss.backward(); optim.step()
assert torch.equal(base.weight, original_weight)
assert base.weight.grad is None
# Store only the adapter; loading it requires the same original base weights.
torch.save({'A': model.A.detach(), 'B': model.B.detach(), 'scale': model.scale}, 'lora-adapter.pt')
state = torch.load('lora-adapter.pt', weights_only=True)
restored_base = nn.Linear(16,12,bias=False)
with torch.no_grad(): restored_base.weight.copy_(original_weight)
restored = LoRALinear(restored_base)
with torch.no_grad():
    restored.A.copy_(state['A']); restored.B.copy_(state['B'])
restored.scale = state['scale']
merged = restored.merged()
assert torch.allclose(model(x_test), merged(x_test), atol=1e-6)
after = F.mse_loss(merged(x_test), y_test).item()
print('Held-out MSE before / after:', before, after)
print('Trainable / full matrix parameters:', model.A.numel()+model.B.numel(), base.weight.numel())
assert after < before * .01
