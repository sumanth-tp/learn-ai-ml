---
id: cv-image-classification
title: "Computer Vision · Sessions 9–10; Image Classification"
sidebar_label: "1 · Image classification"
sidebar_position: 1
slug: /theory/cv/image-classification
description: "Trace pixels to class scores, compute softmax and precision, recall and F1, and design a reliable evaluation."
tags: [computer-vision, classification, softmax, evaluation]
---

import Infographic from '@site/src/components/Infographic';
import ClassificationMetricsLab from '@site/src/components/viz/ClassificationMetricsLab';

**In one line.** Classification assigns an image-level label from learned visual evidence, and evaluation asks which errors that decision makes.

:::tip Before you start
**You should already know**

- How a linear model turns numbers into class scores: [classification and logistic regression](/docs/theory/ml/classification-and-logistic-regression).
- How precision, recall and F1 are read: [model evaluation](/docs/theory/ml/model-evaluation).
- What HoG features are: [Harris corners and HoG](/docs/theory/cv/harris-corners-and-hog).

**Reading time.** About 50 minutes, plus about 80 seconds to run the large experiment (it downloads 30 MB once).

**After this chapter you can**

- turn logits into probabilities and a confusion count into precision, recall and F1, by hand,
- compare a hand-built feature pipeline, a small network and a pretrained network on equal footing, with learning curves,
- say which kind of change in the test images each pipeline survives, using measured numbers.
:::

## In 30 seconds

A classifier looks at a picture and says which label fits best. A photograph of a shirt is just a grid of brightness numbers, and the same shirt can be shifted, dimmer or grainier, so the numbers change while the label does not. You can feed those numbers straight into a linear model, summarise them first with hand-built features such as HoG, train a small network, or reuse a network that someone trained on a million photographs. In this chapter all four are given the same few hundred images, and then the test images are changed slightly. The one that wins on clean images is not the one that wins on changed images.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Logit | A raw class score before normalising | 2.0, 1.0, 0.0 |
| Softmax | Turns scores into probabilities that add to one | 2, 1, 0 become 0.665, 0.245, 0.090 |
| Precision | Of the images flagged, the share that were right | 40 of 50 is 0.80 |
| Recall | Of the images that should be flagged, the share found | 40 of 60 is 0.667 |
| F1 | A single score from precision and recall | 0.727 |
| Learning curve | Accuracy as the training set grows | 0.660 at 50 images, 0.779 at 500 |
| Transfer learning | Reusing features learned on one task for another | ResNet18 trained on ImageNet |
| Distribution shift | Test images that differ from the training images | A 3 pixel shift |

## The idea in plain words

:::note Added to the course material

The decision-threshold analysis, failure cases, runnable checks and design discussion go beyond the course notes. The semantic-gap outline, model sequence, worked examples and all five practice questions remain below.

:::

A photograph is a grid of measured intensities, while a class label is a statement about an object or scene. The same object can occupy different positions, be viewed from different angles, appear under different lighting and be partly hidden. Conversely, two classes may look similar at the resolution available. This mismatch is called the **semantic gap**. An image classifier learns a rule from labelled examples that maps the observed pixels to class scores. It cannot infer a stable label merely because pixels have a particular brightness or an edge appears in one place. The training examples and evaluation set must represent the changes the deployed system will encounter.

The usual model sequence illustrates successive ways to make that rule. A nearest-neighbour classifier compares the new image representation with saved examples, so the representation and distance measure determine which examples look close. A linear classifier maps a feature vector to class scores using a weighted sum. A convolutional neural network learns local filters and combines their responses over a larger receptive field. A vision transformer divides an image into patches and combines their representations using attention. These are architecture families, not an automatic ranking of quality. The right comparison holds the dataset, input resolution, compute budget and evaluation procedure fixed.

The deep-learning details of convolution, pretrained backbones and transfer learning are developed in the existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is), [pretrained CNN chapter](/docs/theory/dnn/pretrained-cnn-models-and-imagenet) and [transfer-learning chapter](/docs/theory/dnn/transfer-learning-feature-extraction-vs-fine-tuning). Here the focus is the computer-vision decision: what is being classified, how scores become predictions and how to measure useful performance. An image-level label cannot tell an application where the object is. If the task requires a location or pixel boundary, detection or segmentation is a different output contract.

A model's logits are unrestricted class scores. Softmax exponentiates and normalises them to nonnegative values summing to one. Subtracting the largest logit before exponentiation avoids avoidable overflow without changing the result. For logits $[2,1,0]$, the probabilities round to $[0.665,0.245,0.090]$. A high softmax output is not by itself proof that a model is calibrated or that an image belongs to one of its known classes. Softmax only compares the supplied classes under this model. Distribution shift and unknown classes need separate evaluation.

<Infographic src="/img/cv/classification.svg" alt="Image classification board showing logits two, one and zero; softmax probabilities point six six five, point two four five and point zero nine zero; and precision, recall and F1 from forty true positives, ten false positives and twenty false negatives." caption="An individual score vector and an aggregate error measure are different pieces of evidence." />

## Worked example, step by step

**From pixels to probabilities.** Take a tiny image of two numbers, $x=(1,2)$, and three classes with weight rows $(1,0)$, $(0,1)$ and $(-1,-1)$ and no bias.

1. Each class score is a weighted sum: $1\times1+0\times2=1$, $0\times1+1\times2=2$ and $-1\times1-1\times2=-3$.
2. Exponentiate: $e^{1}=2.718$, $e^{2}=7.389$ and $e^{-3}=0.0498$. Their sum is 10.157.
3. Divide each by the sum: $0.268$, $0.727$ and $0.005$.

In words: the second class has the largest score and takes most of the probability, but softmax never says how sure the model should be about an image it has never seen.

**From counts to metrics.** On 100 images, 60 are damaged and 40 are fine. The classifier flags 50 and gets 40 of them right, so TP is 40, FP is 10 and FN is 20.

1. Precision is $40/(40+10)=0.80$.
2. Recall is $40/(40+20)=0.667$.
3. F1 is $2\times40/(2\times40+10+20)=80/110=0.727$.

**Why accuracy alone misleads.** Take 99 normal images and 1 damaged one. A classifier that always says "normal" is right 99 times, so accuracy is 0.99, yet its recall for the damaged class is $0/1=0$ and F1 is 0.

## How it works

### The semantic gap

Pixels vary with viewpoint, lighting, deformation, clutter; but the label is constant. Data-driven learning bridges the gap.

### k-NN → CNN → ViT

- **k-NN / linear**; Majority of nearest neighbours; or f=Wx+b with softmax/SVM loss.
- **CNN**; Conv+ReLU, pooling, FC, softmax; learns edge→part→object features.
- **ViT**; Patches + self-attention; hardware-efficient, scales with data.

:::tip

**Worked.** Logits [2,1,0] → softmax [0.665, 0.245, 0.090]. TP=40,FP=10,FN=20 → P=0.80, R=0.667, F1=0.727.

:::

### Evaluation

Accuracy misleads on imbalance. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). Softmax→probabilities; cross-entropy trains them.

### Key takeaways

- **1 · Semantic gap**; Learn label from varying pixels.
- **2 · Models**; k-NN→linear→CNN→ViT.
- **3 · Metrics**; Softmax; precision/recall/F1.

## A real system that works this way

The official Torchvision model catalogue lists image-classification model families and their preprocessing transforms. It is an example of why a deployed classifier must record more than the architecture name. A set of learned weights was trained with particular input sizes, colour conventions and normalisation. Changing a resize or channel order at serving time changes the data the model receives, even if the weight file and the class head are identical. The catalogue and model documentation were opened on 2026-10-02; this chapter does not claim to have benchmarked any of those model families.

Consider a factory line that flags images of damaged packages. The product decision is not necessarily the largest class score. A false negative might allow a damaged package through; a false positive might send a good package for manual inspection. The decision threshold therefore follows a cost and capacity policy. The confusion counts in the worked example, TP 40, FP 10 and FN 20, yield precision 0.8 and recall 2/3 at one such threshold. Increasing the threshold often reduces the number flagged, with a possible gain in precision and loss in recall. The exact trade-off must be measured on held-out examples; those three counts cannot predict what another threshold would do.

The same line may see several camera views, packaging colours and lighting conditions. A random image split can leak near-duplicates from the same product run into both training and validation. Split by the entity or acquisition period that will be new in production, and report metrics separately by camera, product type and lighting condition. Inspect the false negatives as images. Some may reflect missing evidence because the damage is outside the crop; no classifier architecture can recover an unseen defect. Others may show a consistent appearance that belongs in training. That distinction guides whether to change the camera, data, label policy or model.

## Code you can run

The first block computes a softmax with a stable shift. The three printed values reproduce the rounded probabilities of the worked example, and the assertion checks that they sum to one. These values are a mathematical transformation of logits, not measured correctness or calibration.

```python
from math import exp, isclose

logits = [2.0, 1.0, 0.0]
shifted = [exp(value - max(logits)) for value in logits]
probabilities = [value / sum(shifted) for value in shifted]
print('Softmax:', [round(value, 3) for value in probabilities])
assert [round(value, 3) for value in probabilities] == [0.665, 0.245, 0.09]
assert isclose(sum(probabilities), 1.0)
```

The second block computes precision, recall and F1 directly from the same confusion counts. The algebraic F1 form, $2TP/(2TP+FP+FN)$, avoids a second calculation from rounded precision and recall. It produces **0.727** to three decimals. True negatives are not needed for these three metrics, but are needed to calculate accuracy and specificity.

```python
tp, fp, fn = 40, 10, 20
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * tp / (2 * tp + fp + fn)
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}')
assert (round(precision, 3), round(recall, 3), round(f1, 3)) == (0.8, 0.667, 0.727)
```

The lab begins with both of these worked examples. Move the first logit to see the single-image probabilities change; move TP, FP and FN to see the aggregate metrics change. They are separate controls because a single example's logits do not determine a dataset's confusion matrix. The table view exposes every value without relying on the lengths or colours of the bars.

<ClassificationMetricsLab />

**What each control does.**

- *First logit* moves the first class score while the other two stay at 1 and 0.
- *True positives*, *False positives* and *False negatives* set the confusion counts.
- The table shows the three probabilities and the precision, recall and F1.

**Try it yourself.**

1. Leave the defaults: probabilities 0.665, 0.245 and 0.090, precision 0.800, recall 0.667 and F1 0.727.
2. Set the first logit to 4. The first probability rises to 0.936, since $e^4/(e^4+e+1)=54.6/58.3$. Raising a score has diminishing returns because the probabilities cannot exceed 1.
3. Set false negatives to 0 and keep the rest. Recall becomes 1.000 and precision stays 0.800, so F1 rises to 0.889. Finding every damaged package does not make the flags more accurate.

If a selected class has no predicted positives, its precision denominator is zero; if there are no actual positives, its recall denominator is zero. An evaluation report must state its convention for undefined metrics. Setting them silently to one would reward a classifier that never predicts a rare class. The lab calls them undefined when the relevant denominator is zero. In a multiclass problem, define whether counts are per class, micro-averaged or macro-averaged before interpreting one F1 number.

The next block reproduces the worked example and the accuracy trap, using scikit-learn's metric functions.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

x = np.array([1.0, 2.0])
weights = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
scores = weights @ x
exponentials = np.exp(scores - scores.max())
print('scores', scores.tolist(), 'probabilities', (exponentials / exponentials.sum()).round(3).tolist())

truth = np.array([1] * 60 + [0] * 40)
predicted = np.array([1] * 40 + [0] * 20 + [1] * 10 + [0] * 30)
precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average='binary')
print(f'TP 40 FP 10 FN 20: precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}')

truth = np.array([0] * 99 + [1])
always_normal = np.zeros(100, dtype=int)
precision, recall, f1, _ = precision_recall_fscore_support(truth, always_normal, average='binary', zero_division=0)
print(f'always "normal": accuracy {accuracy_score(truth, always_normal):.2f}, recall {recall:.2f}, F1 {f1:.2f}')
```

**Reading the output.** The first line prints the scores and the probabilities 0.268, 0.727 and 0.005. The second reproduces precision 0.800, recall 0.667 and F1 0.727 from labelled arrays. The third shows an always-normal classifier with accuracy 0.99 and recall 0.00.

### Experiment: four pipelines, the same few hundred images

The data is Fashion-MNIST: 28 by 28 grey images of ten clothing types, released under the MIT licence (Hugging Face dataset card, opened 2026-10-09; Xiao, Rasul and Vollgraf, arXiv 1708.07747, 2017). The experiment draws 5, 10, 20 or 50 training images per class (50 to 500 images) three times, tests on 2,000 held-out test images (200 per class), and compares:

- raw pixels with logistic regression,
- HoG features (1,296 values) with a linear SVM,
- a small convolutional network trained from scratch (20,490 weights, 40 epochs),
- the pretrained ResNet18 of torchvision (ImageNet weights, 512 features after the last pooling) with logistic regression on top.

After the learning curve, the models trained on 500 images are tested again on three changed copies of the test set: shifted by 3 pixels, contrast scaled to 40% with a lift, and Gaussian noise of 0.2. The block downloads the dataset once into the temporary folder and loads the cached ResNet18 weights. It took about 75 seconds on a 4-thread CPU here.

```python
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import shift as shift_image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchvision import datasets, models

torch.set_num_threads(4)
root = os.path.join(tempfile.gettempdir(), 'fashion-mnist')
train_set = datasets.FashionMNIST(root, train=True, download=True)
test_set = datasets.FashionMNIST(root, train=False, download=True)
x_train, y_train = train_set.data.numpy().astype(np.float32) / 255, train_set.targets.numpy()
rng = np.random.default_rng(0)
pool = np.concatenate([rng.choice(np.flatnonzero(test_set.targets.numpy() == c), 200, replace=False) for c in range(10)])
x_test, y_test = test_set.data.numpy()[pool].astype(np.float32) / 255, test_set.targets.numpy()[pool]
variants = {
    'clean': x_test,
    'shift 3 px': np.array([shift_image(img, (3, 3), order=1, mode='constant') for img in x_test]),
    'contrast x0.4': np.clip(x_test * 0.4 + 0.3, 0, 1).astype(np.float32),
    'noise 0.2': np.clip(x_test + np.random.default_rng(5).normal(0, 0.2, x_test.shape), 0, 1).astype(np.float32),
}
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
backbone.fc = nn.Identity()
mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def deep(batch):
    with torch.no_grad():
        chunks = [torch.from_numpy(batch[i:i + 250])[:, None].repeat(1, 3, 1, 1) for i in range(0, len(batch), 250)]
        return np.concatenate([backbone((F.interpolate(c, size=112, mode='bilinear') - mean) / std).numpy() for c in chunks])


def hog_all(batch):
    return np.array([hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys') for img in batch])


def cnn(images, labels, seed, epochs=40):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    data, target = torch.from_numpy(images)[:, None], torch.from_numpy(labels)
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for start in range(0, len(data), 32):
            batch = order[start:start + 32]
            optimiser.zero_grad()
            F.cross_entropy(net(data[batch]), target[batch]).backward()
            optimiser.step()
    return net, sum(p.numel() for p in net.parameters())


test_hog = {name: hog_all(images) for name, images in variants.items()}
test_deep = {name: deep(images) for name, images in variants.items()}
print('HoG dimensions', test_hog['clean'].shape[1], '| ResNet18 feature dimensions', test_deep['clean'].shape[1])
curve, shifted = {}, {}
for per_class in (5, 10, 20, 50):
    for seed in range(3):
        r = np.random.default_rng(seed)
        pick = np.concatenate([r.choice(np.flatnonzero(y_train == c), per_class, replace=False) for c in range(10)])
        xs, ys = x_train[pick], y_train[pick]
        raw = LogisticRegression(max_iter=2000).fit(xs.reshape(len(xs), -1), ys)
        svm = LinearSVC(C=1.0, max_iter=20000, random_state=0).fit(hog_all(xs), ys)
        net, params = cnn(xs, ys, seed)
        head = LogisticRegression(max_iter=3000).fit(deep(xs), ys)
        for name in variants:
            with torch.no_grad():
                cnn_pred = net(torch.from_numpy(variants[name])[:, None]).argmax(1).numpy()
            scores = (raw.score(variants[name].reshape(len(y_test), -1), y_test), svm.score(test_hog[name], y_test), (cnn_pred == y_test).mean(), head.score(test_deep[name], y_test))
            curve.setdefault((per_class, name), []).append(scores)
labels = ('raw pixels + logistic', 'HoG + linear SVM', 'small CNN', 'ResNet18 features + logistic')
print('CNN parameters:', params)
print('clean test accuracy by training size (mean of 3 draws)')
print(f'{"images":>8}  ' + '  '.join(f'{label:>28}' for label in labels))
for per_class in (5, 10, 20, 50):
    runs = np.array(curve[(per_class, 'clean')])
    print(f'{per_class * 10:>8}  ' + '  '.join(f'{mean:>22.3f} +/- {sd:.3f}' for mean, sd in zip(runs.mean(axis=0), runs.std(axis=0))))
print('trained on 500 images, tested on changed images')
print(f'{"test set":>14}  ' + '  '.join(f'{label:>28}' for label in labels))
for name in variants:
    print(f'{name:>14}  ' + '  '.join(f'{value:>28.3f}' for value in np.mean(curve[(50, name)], axis=0)))
```

**Reading the output.** The first table gives clean test accuracy by training size, as mean plus or minus standard deviation over three random draws of the training set. The second table gives accuracy at 500 training images on four test sets. The first row of the second table equals the last row of the first, as it should.

**Line by line.**

- `deep` upsamples each 28 by 28 image to 112 by 112, repeats the grey channel three times and normalises it as ImageNet images are, then returns the 512 numbers before ResNet18's classification layer.
- `cnn` is two convolution, ReLU and pooling stages and one linear layer, 20,490 parameters in all.
- The changed test sets are built once in `variants` and shared by all four pipelines, so every pipeline is tested on identical images.
- `np.random.default_rng(5).normal(...)` fixes the noise, so reruns match.

#### Reading the experiment

On clean images the four are close. At 500 training images the accuracies are 0.779, 0.791, 0.797 and 0.787, a spread of 0.018, and the standard deviations over draws are 0.002 to 0.011. At 50 images the spread is 0.021 with standard deviations of 0.021 to 0.030, so no ranking is justified. The 20,490-weight network matches a pretrained network built from 11.2 million weights, and plain logistic regression on raw pixels is within 2 points of everything.

The shifted and altered columns break that tie. A 3 pixel shift drops raw pixels to 0.287, the network to 0.288 and HoG to 0.348, while pretrained features keep 0.717. A contrast change leaves HoG at 0.791 and drops raw pixels to 0.365. Noise of 0.2 is the reverse: raw pixels hold 0.758 and the network 0.741, while HoG falls to 0.339 and pretrained features to 0.289.

The surprise is twofold. On this data, transfer learning gave no advantage on clean images, contradicting the common belief that it always helps with a few hundred examples. Its value showed only under shift. And it is brittle to noise, which the grey 28 by 28 upsampled inputs may make worse.

Limits: one dataset of centred, uncluttered, low-resolution images; three training draws; untuned regularisation (`C` was left at 1.0); a short network with no augmentation; ResNet18 sees a small grey picture it was never trained on. A colour photograph task, or augmentation for the network, would change the margins.

<Infographic src="/img/cv-enrich/v2-classification-shifts.svg" alt="Left: clean test accuracy by training size for four pipelines, all between 0.646 and 0.797. Right: accuracy of the same pipelines on shifted, low-contrast and noisy test images." caption="Look at the right table: each pipeline wins a different column, and only the clean row looks alike." />

## Designing with it

Define the unit of prediction. A classifier might label an entire image, a crop around a known object or a frame from a video. If the frame contains three objects, one image-level class can be ambiguous. Document whether multiple labels are allowed and how “unknown”, “unclear” and “no target” are represented. Label-policy ambiguity is often larger than a change between two model architectures. Review disagreements with domain experts and keep a small set of adjudicated examples for regression checks.

Choose a split that matches use. If future items come from new production lots, splitting individual images randomly can overstate generalisation because neighbouring frames and near-identical packages leak into both sets. Hold out acquisition sessions, devices or time periods as appropriate. Deduplicate before splitting and record how many images, entities and classes remain in each partition. A headline score without these details is hard to trust. A small rare class may have too few held-out cases for a stable estimate, so show counts and uncertainty rather than only three decimal places.

Pick metrics around the cost of errors. Accuracy counts all decisions equally and can look high when the majority class dominates. Precision asks how often a positive prediction is right, while recall asks how many actual positives were found. F1 balances the two by their harmonic mean but does not encode the asymmetric cost of a missed defect versus an unnecessary inspection. If a product has an explicit review capacity, measure how many examples can be sent to people per day and select a threshold on a validation set under that capacity. Report performance at the selected threshold on a separate test set.

Check calibration before using score magnitudes as risk estimates. Softmax outputs always sum to one even for nonsense inputs. A reliability diagram or held-out calibration measure can reveal whether predictions at a stated confidence level are correct at a corresponding rate. Calibration can drift when the camera or object mix changes. Track score distributions, class prevalence, human overrides and delayed ground truth. A model that still produces scores and labels can be silently failing if input quality changes.

Plan a fallback for uncertain or unfamiliar images. A low maximum score may be useful as one signal, but it is not a guaranteed detector of unknown classes. Consider explicit quality checks for blur, obstruction and invalid crops, and a review path for uncertain cases. The fallback must be tested in the full workflow, including operator load and latency. A classifier that works in an offline notebook may cause a queue of unreviewed images when used at line speed.

Finally, audit the preprocessing path. Record colour channel order, resize rule, crop, interpolation, intensity scaling and normalisation with the model artifact. Test the same example through training and serving paths and compare tensors before the model. Even a simple RGB/BGR mismatch can shift every class score. A model-family comparison is useful only after the input contract and evaluation protocol are stable.

## Where this stands in 2026

:::info Industry view

- Current Torchvision documentation, opened on 2026-10-02, provides several classification model families with weight-specific preprocessing. No speed or accuracy ranking is asserted here.
- ViTs are often described as hardware-efficient. That is workload and implementation dependent; attention, token count and memory traffic can make a particular ViT more or less efficient than a particular CNN on a specific device.
- The two Python blocks use standard-library arithmetic and were run locally. No image dataset, trained weights or factory-line deployment was used for the examples.

:::

## Common mistakes

1. **Picking a model from the clean test set.** It is the number everyone reports. Here four pipelines scored 0.779 to 0.797 on clean data, and 0.287 to 0.717 after a 3 pixel shift. Test the changes you expect in production.
2. **Assuming a pretrained network always wins on small data.** It feels like free accuracy. On clean images it matched raw pixels (0.787 against 0.779) and on noisy images it collapsed (0.289). Run the cheap baseline first.
3. **Reporting accuracy on an imbalanced set.** 99 normal images and one damaged one give 0.99 accuracy to a classifier that finds nothing. Report recall for the rare class.
4. **Comparing at different input contracts.** A resize or normalisation that differs between pipelines moves scores more than the architecture. Fix the data, resolution and metric first.
5. **Reading a 0.01 gap as a ranking.** With three draws the standard deviation at 500 images was up to 0.011. Say "tied" when the gap is inside it.

## Practice questions

<details>
<summary><strong>Q1.</strong> What is the semantic gap in image classification?</summary>

The mismatch between low-level pixels (which vary with viewpoint, lighting, deformation, clutter) and the constant high-level label; data-driven learning bridges it.<br /><em>Sessions 9-10 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast k-NN, linear classifiers and CNNs.</summary>

k-NN: majority label of nearest neighbours (no training, slow test). Linear: f=Wx+b with softmax/SVM loss. CNN: learns a conv/pool feature hierarchy end-to-end.<br /><em>Sessions 9-10 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Compute softmax of logits [2, 1, 0].</summary>

e²=7.389, e¹=2.718, e⁰=1, sum=11.107 → [0.665, 0.245, 0.090].<br /><em>Sessions 9-10 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> TP=40, FP=10, FN=20. Compute precision, recall and F1.</summary>

Precision=40/50=0.80, recall=40/60=0.667, F1=2(0.8)(0.667)/(1.467)=0.727.<br /><em>Sessions 9-10 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why can accuracy be misleading, and what is a ViT?</summary>

Accuracy misleads on imbalanced data (a majority-class predictor scores high). A Vision Transformer splits the image into patches and uses self-attention; hardware-efficient and scalable.<br /><em>Sessions 9-10 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> Easy. Scores (1, 2, -3). Which class wins, and what is its softmax probability?</summary>

Class 2 wins. The exponentials are 2.718, 7.389 and 0.0498, summing to 10.157, so its probability is $7.389/10.157=0.727$.<br /><em>Easy · numeric</em>

</details>

<details>
<summary><strong>Q7.</strong> Medium. At 500 training images the four pipelines scored 0.779 to 0.797 on clean data. Which would you ship if test images may be shifted, and why?</summary>

The pretrained-feature pipeline: after a 3 pixel shift it kept 0.717 while the others fell to 0.287 to 0.348. If the images may instead be noisy, it is the worst choice (0.289), so the decision depends on which change is expected.<br /><em>Medium · interpretation</em>

</details>

<details>
<summary><strong>Q8.</strong> Stretch. HoG stayed at 0.791 under the contrast change but fell to 0.339 under noise. Using the block-normalisation idea from the Harris and HoG chapter, explain both.</summary>

Block normalisation divides each block by its own length, so scaling the contrast cancels exactly. Noise adds random gradients of its own, which change both the orientation histograms and the block lengths, so there is nothing for normalisation to cancel.<br /><em>Stretch · interpretation</em>

</details>

## Further reading

- [Torchvision classification models](https://docs.pytorch.org/vision/stable/models.html#classification) for current model-family and weights documentation.
- [Stanford CS231n notes](https://cs231n.stanford.edu/) for classifier objectives and visual-recognition foundations.
- Built from the course lecture "cv-s9-10-image-classification" (Lecture Library series).

- Xiao, Rasul and Vollgraf, "Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms", arXiv 1708.07747, 2017 (bibliographic record checked 2026-10-09). The dataset card on Hugging Face (opened 2026-10-09) states the MIT licence, 60,000 training and 10,000 test images of 28 by 28 pixels.
- [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) (opened 2026-10-09): weights `IMAGENET1K_V1`, 11,689,512 parameters, 69.758% top-1 on ImageNet. The page states no licence for the weights, and I did not verify their terms of use.
- Library versions run for the experiment: torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1, scikit-image 0.26.0, NumPy 2.5.3, SciPy 1.18.1.

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source claim

ViTs are often called “hardware-efficient” and scalable. The scalability description concerns how the architecture can use data and compute, but hardware efficiency is not an intrinsic guarantee. Compare a specific CNN and ViT at the same task quality, input size, device, batch size and latency target before drawing an efficiency conclusion. The model sequence is retained above.

:::

## Reading a classification report

Suppose a report says “99% accuracy” on a set with 99 normal images and one damaged image. Predicting normal every time achieves that accuracy while finding no damage. The report needs class prevalence, a confusion matrix, recall for the damaged class and the cost of a missed case. When there is only one positive test example, recall is either zero or one and is extremely unstable. A test with more independent positive cases is needed before shipping a threshold. The number 99% here is a counterexample constructed from counts, not an empirical benchmark for any named model.

A different report gives precision 0.8, recall 0.667 and F1 0.727. Those values are internally consistent with the TP, FP and FN counts of the worked example, but they describe only the evaluated set and threshold. They say nothing about whether false positives cluster in a specific shift, whether all false negatives are tiny defects, or how many true negatives were present. Ask for the confusion matrix by slice, sample images and the threshold selection procedure. If a system exposes softmax outputs, ask whether they have been calibrated on data separated from training and whether the current inputs resemble that data.

Look for the annotation unit. A package with two defects may appear once in image-level counts, twice in object-level counts, or as several pixel regions in segmentation. Mixing these evaluation units can produce conflicting metrics without any arithmetic error. The unit must match the action the product takes: reject a package, draw a box for an inspector, or estimate defect area. Image classification is a good choice when the decision is genuinely global and enough evidence is visible in a standardised view.

Also distinguish validation from monitoring. During development, ground-truth labels permit precision and recall measurement. During live use, ground truth may arrive late or only for inspected cases. A dashboard of average softmax score is not a substitute for sampled labelled audits. Monitor the capture pipeline and review a designed sample of outputs to estimate ongoing errors. Otherwise a drift in lighting or crop can make the score distribution look confident while actual recall falls.

When comparing model families, include the full inference pipeline. Input resize and normalisation, CPU-to-device transfer, batch size and postprocessing can dominate latency. A model that runs quickly on a benchmark GPU may be slow on the target edge device. Conversely, an architecture with fewer parameters is not guaranteed to have lower latency if its operations map poorly to that hardware. State the measurement setup alongside any speed claim. The course outline names architecture families; it does not provide a fair deployment benchmark, so this chapter does not invent one.

Finally, inspect what the classifier learned. If damage examples were all photographed on a blue mat and normal examples on a grey mat, a high test score after a random split may reflect the mat rather than damage. Test counterexamples where the background is swapped or acquisition changes. Evaluate under occlusion and low light, and check whether the crop contains the feature that supports the label. Such tests connect the semantic gap back to data collection: the model receives pixels, so any repeated pixel cue can become its shortcut.

## Check yourself

- I can explain the semantic gap without assuming a fixed object appearance.
- I can turn logits into a stable softmax calculation and state what it does not prove.
- I can compute precision, recall and F1 from TP, FP and FN and handle zero denominators.
- I can choose a split and metric that match an image-level product decision.
- I can compute a softmax from linear scores and a precision, recall and F1 from counts, by hand.
- I can read a learning curve and say when differences are inside the draw-to-draw spread.
- I can say which of raw pixels, HoG, a small network and pretrained features survives a shift, a contrast change and noise, with the measured numbers.
