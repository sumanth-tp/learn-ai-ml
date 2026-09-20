---
id: paper-resnet
title: "Deep Residual Learning for Image Recognition"
sidebar_label: "13 · ResNet"
sidebar_position: 13
slug: /research-papers/resnet
description: "Degradation, residual learning, gradients, shortcut projections, bottlenecks, experiments and a complete trainable convolutional ResNet."
tags: [research-papers, deep-learning]
---

import PaperPdf from '@site/src/components/PaperPdf';

> **He et al. · 2015** · [Read the embedded paper](#original-paper) · [Download PDF](/papers/research-papers/resnet.pdf)


ResNet makes a deep network learn changes to a representation while providing a shortcut that carries the existing representation forward.

## Section 1: the surprising problem with adding layers

A deeper network should be able to imitate a shallower one by making the extra layers behave like identity mappings. Yet the paper shows cases where increasing depth makes **training error** worse.

That distinction matters. If training error is low but test error rises, we suspect overfitting. If training error itself rises with added depth, optimisation has become harder. The paper calls this the degradation problem.

Vanishing gradients are relevant to deep networks, but “ResNet solves vanishing gradients” is too narrow an account. The paper's central question is how to make a useful near-identity mapping easy to represent and optimise.

## Section 3: learn the residual instead of the whole mapping

Let the desired mapping be H(x). A plain block directly approximates H. A residual block learns:

$$
F(x)=H(x)-x,\qquad H(x)=F(x)+x.
$$

Suppose the best output is already close to x. The residual branch only needs to learn a small correction. If the best mapping is exactly identity, a zero residual is enough before the block's final activation.

![A residual block and its identity shortcut](/img/research-papers/resnet.png)

*Figure 2 from the original paper, PDF page 2. [Source PDF](/papers/research-papers/resnet.pdf#page=2).*

The original block applies two convolutional transformations on the residual branch and adds the input through a shortcut. Its post-addition ReLU means the full block is $\operatorname{ReLU}(x+F(x))$. Exact identity therefore needs attention to the activation and input sign; it is not true for arbitrary negative inputs after a ReLU.

### Why the shortcut helps gradients

Before the final activation, the Jacobian is:

$$
\frac{\partial(x+F(x))}{\partial x}=I+\frac{\partial F(x)}{\partial x}.
$$

The identity term supplies a direct gradient route. Without the shortcut, gradients must pass entirely through the residual transformation. This improves the available paths for optimisation, but does not mathematically guarantee that no gradient can ever vanish or explode. Activations and the rest of the network still matter.

## Shapes: addition is stricter than concatenation

To add two tensors, their dimensions must agree. A shortcut from 16 channels at `32 × 32` resolution cannot be directly added to a branch producing 32 channels at `16 × 16` resolution.

A projection shortcut fixes that:

$$
y=F(x)+W_sx.
$$

A `1 × 1` convolution changes channel count, and a stride can reduce spatial resolution. When shapes already agree, an identity shortcut has no parameters.

| Operation | Spatial size | Channels | Role |
|---|---|---|---|
| Identity shortcut | Preserved | Preserved | Carry input directly |
| 1×1 projection, stride 1 | Preserved | Can change | Match channels |
| 1×1 projection, stride 2 | Reduced | Can change | Match downsampling branch |
| Concatenation | Must match non-concatenated axes | Usually increases | Different operation; not residual addition |

The paper compares shortcut options, including parameter-free ways of handling changed dimensions. The example uses learned projections where needed.

## Basic blocks and bottlenecks

A **basic block** uses two 3×3 convolutions. A **bottleneck block** uses 1×1, 3×3 and 1×1 convolutions. The first reduces width, the middle performs spatial processing, and the last expands width again.

Why do this? A 3×3 convolution across many channels is expensive. Performing it at a narrower intermediate width saves computation while the surrounding projections allow a wide residual stream.

Batch normalisation appears within the branch and helps stabilise training. It uses batch statistics during training and stored running statistics during evaluation, so calling `model.eval()` changes behaviour even though weights do not change.

### From blocks to a classifier

The network groups blocks into stages. Later stages reduce spatial resolution and increase channels. Global average pooling reduces each channel to one value, then a fully connected layer produces class logits.

Pooling means the classification head does not need a separate weight for every spatial location. Earlier convolutions still learn spatial features; global pooling does not mean location never mattered.

## Real-world uses and worked examples

### Documented implementation: object detection with a ResNet backbone

Torchvision provides a Faster R-CNN detector with a ResNet-50 and Feature Pyramid Network backbone. ResNet extracts image features; the detection components identify and classify object regions. This is a concrete example of reusing residual representations for a task beyond whole-image classification. [Torchvision's detector documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

### Worked example: count packages in a loading area

A camera image can contain several overlapping packages at different sizes. In an illustrative detection system:

1. A ResNet backbone converts pixels into feature maps.
2. A feature pyramid supplies representations at multiple spatial scales.
3. Detection heads predict package boxes and class scores.
4. Post-processing removes duplicate detections before the application counts boxes.

A plain ResNet classifier would instead produce one set of class scores for the whole image. That alone cannot tell the application where each package is or how many are present. This distinction explains why the detector needs more than a strong backbone.

### Another application: visual quality inspection

A manufacturer could start from pre-trained residual-network features and fine-tune a classifier on labelled images of acceptable and defective components. If the task needs the exact defect location, a detector or segmentation head is more appropriate than an image-level label.

The factory example is illustrative, not a reported named deployment. Lighting, camera angle, rare defect types and production changes must be represented in evaluation. Residual shortcuts make the network easier to optimise; they do not guarantee robustness to a new camera or an unseen defect.

**Connection to the paper:** the learned residual representation is reusable. The downstream output head and training labels determine whether the system classifies an image, finds objects or marks pixels.

## Complete code: train a convolutional residual network

**Teaching implementation.** This program includes basic blocks, bottlenecks, projection shortcuts, batch normalisation, global pooling, training and held-out evaluation on generated stripe images.

Save as `resnet.py`, install PyTorch, and run `python resnet.py`.

```python
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
```

### Follow the shapes through the model

An input batch has shape **B × 1 × 16 × 16**. The stem maps it to 8 channels. The first basic block preserves the shape. The second basic block uses stride 2, giving **B × 16 × 8 × 8**, so its shortcut must project and downsample too.

The bottleneck produces **B × 32 × 4 × 4**. Global average pooling yields one value per channel. The final linear layer produces two logits, corresponding to horizontal and vertical bars.

Training images vary bar position and noise. Evaluation draws fresh examples and uses stored batch-normalisation statistics. The checked run achieved full accuracy on that simple held-out distribution. This is a test of the complete training pipeline, not an ImageNet reproduction or evidence that any residual architecture will generalise perfectly.

The bottleneck places its downsampling stride in the first 1×1 convolution, following the original ImageNet bottleneck recipe. Some later implementations move that stride into the 3×3 convolution. Always check the specific architecture before loading a checkpoint: a familiar model name does not establish identical layer behaviour.

## Section 4: experiments that support the argument

The paper compares plain and residual networks at different depths on ImageNet and CIFAR-10, analyses shortcut choices, and transfers features to detection tasks. It demonstrates that residual formulations can make much deeper networks train effectively. Its widely cited 3.57% ImageNet top-5 test error is an ensemble result, not the score of every individual ResNet.

**Top-1 error** counts whether the highest-scoring class is wrong. **Top-5 error** counts whether the correct class is absent from the five highest-scoring classes. Comparing numbers without their metric can create a false impression of performance.

The CIFAR experiments also show that making a network extremely deep is not automatically beneficial on a small dataset. Better optimisation does not eliminate overfitting or make data and model capacity irrelevant.

## Why this matters beyond computer vision

Transformers also use residual additions around their sublayers. The transferable idea is preserving an existing representation while learning an update. A Transformer block is not a ResNet convolutional block: attention, normalisation order and activation details differ.

The original paper appeared as a 2015 preprint and at CVPR 2016. The [authors' repository](https://github.com/KaimingHe/deep-residual-networks) contains original model material for studying those architectures.

## Network families, shortcut ablations and the depth experiments

### What the number in ResNet-50 counts

The ImageNet architectures organise residual blocks into four stages. Basic blocks contain two main-path convolutions; bottleneck blocks contain three. A stem convolution and final classifier complete the usual named depth count, without counting each shortcut projection as another named layer.

| Model | Block type | Blocks in the four stages |
|---|---|---|
| ResNet-18 | Basic | 2, 2, 2, 2 |
| ResNet-34 | Basic | 3, 4, 6, 3 |
| ResNet-50 | Bottleneck | 3, 4, 6, 3 |
| ResNet-101 | Bottleneck | 3, 4, 23, 3 |
| ResNet-152 | Bottleneck | 3, 8, 36, 3 |

For ResNet-50, sixteen bottleneck blocks give 48 main-path convolutions. Adding the stem and classifier gives the name “50”. The teaching script mixes a few basic and bottleneck blocks for visibility; it is not a standard ResNet-50 checkpoint definition.

### Options A, B and C test the shortcut itself

The original comparison studies three shortcut choices:

- **A:** parameter-free shortcuts, using downsampling and zero-padding when dimensions change.
- **B:** projection shortcuts when dimensions increase, identity shortcuts elsewhere.
- **C:** learned projection shortcuts throughout.

All substantially improve over the comparable plain network in the reported experiment. The modest differences among shortcut variants help support the importance of the residual formulation itself, rather than attributing the entire gain to extra projection parameters.

An identity shortcut is not a learned gate. This also distinguishes the simple residual block from related architectures that explicitly learn how much information to pass through a shortcut.

### The training recipe makes the comparison meaningful

The ImageNet recipe includes scale/crop augmentation, horizontal flips, colour augmentation, batch normalisation and SGD with momentum and weight decay. It does not use dropout in the reported setup. Plain and residual counterparts are trained under comparable conditions to test the effect of adding residual connections.

At evaluation, a single crop, multiple crops, multiple scales and an ensemble are different protocols. Averaging predictions across several image views can improve accuracy without changing the trained architecture. Comparing that result with another model's single-view score would confound architecture and evaluation procedure.

### Why the 1,202-layer result matters

The paper shows that a very deep CIFAR-10 residual network can achieve low training error, yet its test error is worse than a shallower residual model. That observation separates two problems:

1. **Optimisation:** can the network fit the training examples?
2. **Generalisation:** does the learned function work on unseen examples?

Residual learning helps the first problem but does not remove the second. Once a network can fit the data, adding more layers can increase capacity without supplying more evidence about unseen inputs.

The paper also examines the magnitudes of residual responses. Small learned corrections are consistent with the motivation of learning changes around an identity path. They do not mean every block is exactly identity or that deep networks can be replaced by a single shortcut.

### Detection and localisation extend the argument

The detection experiments replace the feature extractor inside a larger detection system. Classification asks which class an image contains; detection also asks where objects are; localisation evaluates spatial prediction under its own protocol. Their metrics, such as mean average precision, should not be read as classification accuracy.

The practical result is that improved learned features can help multiple vision tasks. It does not imply that a classification head alone becomes a detector when the backbone gets deeper. [Original paper, Sections 3–4 and detection/localisation appendix](/papers/research-papers/resnet.pdf).

## Summary and self-check

- [ ] I can distinguish optimisation degradation from overfitting.
- [ ] I can derive the residual mapping and its identity gradient term.
- [ ] I can decide when a shortcut needs a projection.
- [ ] I can trace channel and spatial dimensions through basic and bottleneck blocks.
- [ ] I can explain why training and evaluation modes differ with batch normalisation.
- [ ] I can interpret top-1, top-5 and ensemble results correctly.


## Original paper

<PaperPdf slug="resnet" title="Deep Residual Learning for Image Recognition" />
