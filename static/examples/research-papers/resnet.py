"""Train a small convolutional ResNet, including projection shortcuts.
Teaching adaptation: generated 16x16 stripe images and fewer blocks than ResNet-18.
"""
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7);torch.set_num_threads(1)
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.branch=nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channels),nn.ReLU(),nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels))
        self.shortcut=nn.Identity() if stride==1 and in_channels==out_channels else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),nn.BatchNorm2d(out_channels))
    def forward(self,x):return F.relu(self.branch(x)+self.shortcut(x))

class Bottleneck(nn.Module):
    def __init__(self,in_channels,width,stride=1):
        super().__init__();out_channels=4*width
        self.branch=nn.Sequential(nn.Conv2d(in_channels,width,1,stride,bias=False),nn.BatchNorm2d(width),nn.ReLU(),
            nn.Conv2d(width,width,3,1,1,bias=False),nn.BatchNorm2d(width),nn.ReLU(),
            nn.Conv2d(width,out_channels,1,bias=False),nn.BatchNorm2d(out_channels))
        self.shortcut=nn.Identity() if stride==1 and in_channels==out_channels else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),nn.BatchNorm2d(out_channels))
    def forward(self,x):return F.relu(self.branch(x)+self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,8,3,padding=1,bias=False),nn.BatchNorm2d(8),nn.ReLU(),
            BasicBlock(8,8),BasicBlock(8,16,stride=2),Bottleneck(16,8,stride=2),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,2))
    def forward(self,x):return self.layers(x)

def batch(n):
    labels=torch.randint(2,(n,));images=torch.randn(n,1,16,16)*.15
    for i,label in enumerate(labels):
        position=torch.randint(3,12,()).item()
        if label==0:images[i,0,position:position+2,:]+=1
        else:images[i,0,:,position:position+2]+=1
    return images,labels

model=ResNet();optim=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=1e-4)
for step in range(150):
    x,y=batch(32);loss=F.cross_entropy(model(x),y)
    optim.zero_grad();loss.backward();optim.step()
model.eval()
with torch.no_grad():
    x,y=batch(256);accuracy=(model(x).argmax(-1)==y).float().mean()
print('Held-out stripe accuracy:',accuracy.item())
assert accuracy>.95
torch.save(model.state_dict(),'resnet-demo.pt')
