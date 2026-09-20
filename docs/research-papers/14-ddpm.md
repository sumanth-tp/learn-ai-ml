---
id: paper-ddpm
title: "Denoising Diffusion Probabilistic Models"
sidebar_label: "14 · DDPM"
sidebar_position: 14
slug: /research-papers/ddpm
description: "Forward diffusion, variational training, noise prediction, U-Net conditioning, reverse sampling and a complete runnable image diffusion experiment."
tags: [research-papers, deep-learning]
---

import PaperPdf from '@site/src/components/PaperPdf';

> **Ho, Jain and Abbeel · 2020** · [Read the embedded paper](#original-paper) · [Download PDF](/papers/research-papers/ddpm.pdf)


DDPM learns to generate data by reversing a gradual process that turns examples into Gaussian noise.

## Why turn a good image into noise?

Generating an image in one step is a difficult mapping. Diffusion replaces it with many smaller denoising transitions. During training, we can corrupt a real image ourselves, so we know exactly which noise was added.

The learned model receives the corrupted image and its noise level, then predicts a quantity that helps reverse the corruption. At generation time, it starts from fresh noise and repeatedly applies learned reverse transitions.

The forward process is fixed. The reverse process is learned. Keeping those roles separate makes the equations much easier to follow.

## Section 2: define the forward process

Let $x_0$ be a clean example. Choose a variance schedule $\beta_1,\ldots,\beta_T$, with $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$.

One forward step is:

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).
$$

The signal is slightly reduced and Gaussian noise is added. Repeating this enough times produces something close to standard Gaussian noise when the cumulative retained signal becomes small.

![Forward noising and learned reverse transitions](/img/research-papers/ddpm.png)

*Figure 2 from the original paper, PDF page 2. [Source PDF](/papers/research-papers/ddpm.pdf#page=2).*

In the original figure, the noisy endpoint appears on the left and the clean image on the right. Follow the labelled q and p arrows, rather than assuming left-to-right always means forward noising.

### The closed form saves training work

Because the transitions are Gaussian, we can jump directly from clean data to a chosen noise level:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).
$$

This is why training does not need to execute all previous forward steps for every example. Sample t, sample noise, and construct $x_t$ directly.

For a one-dimensional example with $x_0=2$, $\bar\alpha_t=0.64$ and $\epsilon=-1$, the noisy value is `0.8 × 2 + 0.6 × (-1) = 1`. The retained signal and noise coefficients are square roots because the schedule describes variances.

## Section 3: learn a reverse Gaussian transition

The model represents:

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I).
$$

It must know t because the same-looking input requires different treatment at different noise levels. A nearly clean image needs a small correction; a mostly noisy image requires much more inference from the learned data distribution.

The forward posterior $q(x_{t-1}\mid x_t,x_0)$ has a tractable Gaussian form. During training x₀ is available, allowing the reverse distribution to be compared with that posterior. At generation time x₀ is unknown, so the network supplies the missing information.

### Predicting noise is a parameterisation of the reverse mean

DDPM expresses the mean using a predicted noise vector:

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

The model is not subtracting all noise in one arbitrary step. The coefficient depends on the schedule and corresponds to the chosen reverse transition.

The paper discusses fixed reverse-variance choices. The code uses the forward posterior variance:

$$
\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
$$

## The variational objective and the practical loss

The likelihood objective can be bounded by a sum of terms: a prior-matching term at the noisy endpoint, KL terms comparing intermediate transitions, and a reconstruction term at the clean endpoint.

After the noise-prediction parameterisation, intermediate terms correspond to **weighted** noise-prediction errors. The paper's simplified training objective drops those timestep-dependent weights:

$$
L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2\right].
$$

The unweighted MSE is therefore related to the variational derivation, but is not literally every term of the original likelihood bound. This distinction matters when comparing sample-quality optimisation with likelihood evaluation.

### Algorithm 1: training

1. Draw a clean example.
2. Draw a timestep and Gaussian noise.
3. Construct the noisy example using the closed form.
4. Predict its noise using the example and timestep.
5. Backpropagate the squared error.

### Algorithm 2: sampling

Start with $x_T\sim\mathcal N(0,I)$. For decreasing timesteps, calculate the predicted mean and sample:

$$
x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z.
$$

Use fresh Gaussian z except at the final clean-data step, where the added noise is zero. Stochastic sampling produces different outputs from different random starts and transitions.

## The denoiser: why a U-Net and time embeddings?

A U-Net reduces spatial resolution to collect broader context, then upsamples while reusing higher-resolution features through skip connections. These concatenation skips preserve detail; they are different from the residual additions inside individual blocks.

Timestep embeddings turn the scalar noise level into a vector the network can use throughout its layers. The example uses sinusoidal time features and learned projections into convolutional blocks.

The original image model uses a much larger architecture and training setup, including attention at selected resolutions. The narrow teaching U-Net below preserves the training and sampling path but omits that original architectural scale and attention.

## Real-world uses and worked examples

### Documented use: image super-resolution with SR3

Google's SR3 work adapts denoising diffusion probabilistic models to conditional image generation for super-resolution. The model produces a higher-resolution image through iterative denoising while using a lower-resolution image as a condition. This is a direct research extension of the DDPM approach. [The SR3 publication](https://research.google/pubs/image-super-resolution-via-iterative-refinement/).

### Worked example: enlarge a small image

Imagine an application enlarging a small illustration for a presentation. A diffusion-based super-resolution pipeline supplies the low-resolution image to a denoiser, starts a high-resolution sample from noise, and iteratively refines it while remaining conditioned on the input.

The input constrains broad content, but it does not contain every missing pixel. The model generates plausible detail. A sharp-looking letter or facial feature can therefore be invented; higher visual quality is not proof that the original detail has been recovered.

### Another documented extension: Imagen's image-generation cascade

The Imagen research system uses a text-conditioned base diffusion model followed by text-conditioned super-resolution stages. This combines the repeated-denoising idea with language conditioning and a resolution cascade. These additions go beyond the original unconditional DDPM setup. [The Imagen project](https://imagen.research.google/).

A possible design workflow is to request a draft product illustration, inspect generated candidates and refine the prompt. The text representation conditions denoising; the model does not look up a pre-existing photograph corresponding exactly to the sentence.

| Use | Condition supplied to denoising | Output |
|---|---|---|
| Super-resolution | Low-resolution image | A plausible higher-resolution version |
| Text-to-image generation | Text representation | A new image matching the description to some degree |
| Original teaching script here | No external condition | Samples from the learned small image distribution |

**Connection to the paper:** all three learn reverse denoising, while their conditions, model architectures and intended outputs differ.

## Complete code: train a denoiser and generate new images

**Teaching implementation.** This program includes a schedule, direct forward noising, a time-conditioned U-Net, training, the complete reverse loop and saved generated images.

Save as `ddpm.py`, install PyTorch, and run `python ddpm.py`. It writes 16 small `.pgm` images to `ddpm-samples/` and saves a model checkpoint.

```python
"""Complete DDPM training and reverse sampling on small monochrome images.
Teaching adaptation: 8x8 bars, 50 diffusion steps and a narrow time-conditioned U-Net.
Writes samples as PGM images, which can be opened by common image viewers.
"""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7);torch.set_num_threads(1)
T=50
beta=torch.linspace(.0001,.2,T)
alpha=1-beta;alpha_bar=torch.cumprod(alpha,0)
previous=torch.cat((torch.ones(1),alpha_bar[:-1]))
posterior_variance=beta*(1-previous)/(1-alpha_bar)

def extract(values,t):return values[t][:,None,None,None]
def q_sample(clean,t,noise):
    return extract(alpha_bar,t).sqrt()*clean+(1-extract(alpha_bar,t)).sqrt()*noise

def time_embedding(t,width=32):
    frequency=torch.exp(-math.log(10000)*torch.arange(width//2)/(width//2))
    angle=t.float()[:,None]*frequency
    return torch.cat((angle.sin(),angle.cos()),1)

class TimeBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.conv1,self.conv2=nn.Conv2d(cin,cout,3,padding=1),nn.Conv2d(cout,cout,3,padding=1)
        self.norm1,self.norm2=nn.GroupNorm(4,cout),nn.GroupNorm(4,cout)
        self.time=nn.Linear(32,cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,time):
        h=F.silu(self.norm1(self.conv1(x)))+self.time(time)[:,:,None,None]
        h=self.norm2(self.conv2(F.silu(h)))
        return F.silu(h+self.skip(x))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,32))
        self.down=TimeBlock(1,16);self.middle=TimeBlock(16,32)
        self.up=TimeBlock(48,16);self.output=nn.Conv2d(16,1,1)
    def forward(self,x,t):
        time=self.time(time_embedding(t))
        skip=self.down(x,time)
        middle=self.middle(F.avg_pool2d(skip,2),time)
        up=F.interpolate(middle,size=skip.shape[-2:],mode='nearest')
        return self.output(self.up(torch.cat((up,skip),1),time))

def batch(n):
    x=-torch.ones(n,1,8,8)
    for i in range(n):
        position=torch.randint(1,6,()).item()
        if torch.rand(())<.5:x[i,0,position:position+2,:]=1
        else:x[i,0,:,position:position+2]=1
    return x

model=UNet();optim=torch.optim.Adam(model.parameters(),lr=.002)
losses=[]
for step in range(600):
    clean=batch(32);t=torch.randint(T,(32,));noise=torch.randn_like(clean)
    noisy=q_sample(clean,t,noise)
    loss=F.mse_loss(model(noisy,t),noise)
    optim.zero_grad();loss.backward();optim.step();losses.append(loss.item())
model.eval()
with torch.no_grad():
    x=torch.randn(16,1,8,8)
    for step in reversed(range(T)):
        t=torch.full((len(x),),step,dtype=torch.long)
        prediction=model(x,t)
        mean=(x-beta[step]/(1-alpha_bar[step]).sqrt()*prediction)/alpha[step].sqrt()
        x=mean+posterior_variance[step].sqrt()*torch.randn_like(x) if step>0 else mean
    assert torch.isfinite(x).all()
    images=((x.clamp(-1,1)+1)*127.5).byte()
Path('ddpm-samples').mkdir(exist_ok=True)
for i,pixels in enumerate(images[:,0]):
    Path(f'ddpm-samples/{i:02d}.pgm').write_bytes(b'P5\n8 8\n255\n'+bytes(pixels.flatten().tolist()))
print('Mean first / last 50 losses:',sum(losses[:50])/50,sum(losses[-50:])/50)
print('Wrote 16 generated samples to ddpm-samples/')
torch.save(model.state_dict(),'ddpm-demo.pt')
```

### Follow the schedule and indexing

The mathematical paper indexes timesteps from 1 to T. Python arrays use 0 through T−1. `previous` prepends a retained-signal value of 1 so that the posterior-variance formula has the correct clean-data boundary.

`extract` reshapes one schedule value per batch item to **B × 1 × 1 × 1**, allowing it to broadcast across pixels. Without this reshape, a timestep vector can accidentally align with an image dimension.

The training loop samples different t values for different examples. The sampling loop instead moves the whole batch through each decreasing timestep. These are distinct uses of the same denoiser.

The 50-step schedule uses a larger final beta than the paper's longer schedule so that the retained signal becomes small within this short demonstration. Copying the original endpoint while drastically reducing the number of steps would leave too much signal at the supposed noise endpoint.

In the checked run, average noise-prediction loss fell from about 0.30 over the first 50 steps to about 0.045 over the last 50. Inspect the saved samples as well: a falling loss alone does not prove good generative coverage or sharp images.

## Section 4: results and what they measure

The paper evaluates image generation with sample-quality metrics and likelihood-related quantities, compares objective choices, and explores interpolation and progressive reconstruction. These analyses show that denoising models can generate strong image samples, while also revealing trade-offs between quality, compression-style interpretation and likelihood.

**FID** compares statistics of generated and real image features; it is not a direct per-image correctness score. **Likelihood** measures assigned probability under the model and its evaluation procedure. Neither alone captures every aspect of visual quality or memorisation.

DDPM here is a pixel-space generative model. Latent diffusion moves the diffusion process into an autoencoder's latent space; text-conditioned systems add conditioning mechanisms. Those later designs build on related ideas but are not all introduced in this paper. The [authors' implementation](https://github.com/hojonathanho/diffusion) is available for the original experiments.

## The remaining derivation: likelihood, scores and progressive reconstruction

### Write the variational bound as separate terms

A useful form of the negative log-likelihood bound is:

$$
L=\mathbb E_q\left[
D_{\mathrm{KL}}(q(x_T\mid x_0)\Vert p(x_T))
+\sum_{t=2}^{T}D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t))
-\log p_\theta(x_0\mid x_1)
\right].
$$

The first term checks the noisy endpoint against the chosen prior. The middle terms train reverse transitions. The final term assigns likelihood to the actual clean data. With a fixed forward schedule, the prior-matching term does not train the denoiser parameters.

The true forward posterior has variance $\tilde\beta_t$ and mean:

$$
\tilde\mu_t(x_t,x_0)=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

This is a weighted combination of the clean example and its noisy version. Training knows both; generation knows only the noisy version. Substituting the forward noising equation for x₀ leads to the noise-prediction mean parameterisation used earlier.

### Continuous Gaussian outputs versus discrete pixels

The original images contain integer pixel values, scaled into the interval from −1 to 1. A continuous density evaluated at one number is not the same as the probability of a discrete pixel value.

For likelihood evaluation, the paper's final decoder integrates Gaussian probability over the interval corresponding to each pixel bin. Endpoint bins include the appropriate tail. This makes the reconstruction term a probability for discrete image data, supporting the bits-per-dimension interpretation.

The teaching script trains the simplified denoising loss and generates images. It does not calculate this discretised decoder likelihood or report a variational bound. Its MSE is therefore not a substitute for the paper's likelihood metric.

### The connection to score matching

A **score** here means the gradient of log density with respect to a noisy input, not an evaluation score. For Gaussian-corrupted data, noise prediction is related to estimating that direction:

$$
s_\theta(x_t,t)\approx-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.
$$

The score points towards locally more likely noisy-data configurations. This connects denoising diffusion to denoising score matching and stochastic sampling methods such as Langevin dynamics. The schedule and discrete reverse update still matter; the terms should not be treated as interchangeable names for every sampler.

### What the ablations establish

The paper compares predicting the reverse mean with predicting noise, and the full variational objective with the simplified objective. It also studies fixed versus learned reverse variances. Noise prediction with the simplified objective gives strong sample-quality results in that setup; learning the variance was less stable there.

That is a finding about the original implementation. Later work can change variance parameterisation, objectives or sampling methods. The 2020 result is not a rule that reverse variance must never be learned.

### Progressive compression and interpolation

The paper interprets the reverse process as gradually recovering information. At high noise, fine detail is absent; later stages add progressively more detail. Its compression analysis separates a rate-like cost of transmitting latent information from a distortion-like reconstruction cost, helping explain why strong-looking samples and best likelihood need not coincide.

It also corrupts two source images to a selected noise level, interpolates their noisy representations, and runs the reverse process. Changing the corruption level changes how much original detail survives. This is different from simply averaging clean pixels, which can produce a transparent-looking overlay.

The original architecture uses a larger U-Net with residual blocks, group normalisation, timestep embeddings and attention at selected resolutions. Its experiments use many more diffusion steps than our small example. The code covers Algorithm 1-style denoising training and Algorithm 2-style reverse sampling; the likelihood, compression and interpolation analyses are explained here but are separate experiments. [Original paper, Sections 2–5 and Appendices A–C](/papers/research-papers/ddpm.pdf).

## Summary and self-check

- [ ] I can distinguish beta, alpha and cumulative alpha-bar.
- [ ] I can construct a noisy example directly at any timestep.
- [ ] I can explain why predicting noise determines the reverse mean.
- [ ] I can distinguish the simplified MSE from the full variational bound.
- [ ] I can trace the entire training and sampling algorithms through the code.
- [ ] I can explain why the last reverse step adds no fresh noise.
- [ ] I can distinguish pixel diffusion, latent diffusion and text conditioning.


## Original paper

<PaperPdf slug="ddpm" title="Denoising Diffusion Probabilistic Models" />
