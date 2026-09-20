"""Train image/text encoders with symmetric contrastive loss, then classify.
Teaching adaptation: generated stripe images and one-word text labels.
"""
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
labels = ['horizontal', 'vertical', 'diagonal']
def images(n):
    targets = torch.arange(n) % 3
    x = torch.randn(n,1,8,8)*.1
    for i, target in enumerate(targets):
        if target==0: x[i,0,3:5,:] += 1
        elif target==1: x[i,0,:,3:5] += 1
        else: x[i,0].diagonal().add_(1)
    return x,targets

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(512,16))
        self.text_encoder = nn.Embedding(3,16)
        self.log_scale = nn.Parameter(torch.tensor(1/.07).log())
    def encode_image(self,x): return F.normalize(self.image_encoder(x),dim=-1)
    def encode_text(self,t): return F.normalize(self.text_encoder(t),dim=-1)
    def forward(self,x,t): return self.log_scale.exp() * self.encode_image(x) @ self.encode_text(t).T

model = CLIP()
optim = torch.optim.Adam(model.parameters(),lr=.003)
for step in range(200):
    # Exactly one of each class per batch avoids identical labels as false negatives.
    x,t = images(3)
    logits = model(x,t)
    target = torch.arange(3)
    loss = (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
    optim.zero_grad(); loss.backward(); optim.step()
    with torch.no_grad(): model.log_scale.clamp_(max=torch.tensor(100.).log())
model.eval()
with torch.no_grad():
    test,target = images(120)
    # The label text vectors act as classifier weights, with no classifier training.
    class_vectors = model.encode_text(torch.arange(3))
    prediction = (model.encode_image(test) @ class_vectors.T).argmax(-1)
    accuracy = (prediction==target).float().mean()
print('Held-out image accuracy:',accuracy.item())
print('First predictions:',[labels[i] for i in prediction[:6]])
assert accuracy > .95
torch.save(model.state_dict(),'clip-demo.pt')
# These label words were seen in training: this checks the zero-shot classifier
# construction, not transfer to unseen natural-language concepts.
