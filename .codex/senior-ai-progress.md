# Senior AI curriculum: progress and source ledger

Both agents update this file. Add rows; do not rewrite other people's. Dates are absolute.

## Phase status

| Phase | State | Updated |
| --- | --- | --- |
| 0 Foundation | Plan, brief, ledger written. Venv rebuilt. bansal ML, IR, DM, DML, CV crawled (84 pages) and converted cleanly. Decisions confirmed by the user. Stage scaffolding done in code (10 stages, "Coming soon" for empty ones), not yet built. | 2026-10-01 |
| 1 P0 core | Claude: Track A finished and independently verified in the browser; Track B authors active. Codex: B1 16/16 and C1 18/18 complete. P1 open; H started with CV sources converted. | 2026-10-02 |

## Tracks

| Track | Owner | State | Chapters done / planned | Cross-reviewed | Notes |
| --- | --- | --- | --- | --- | --- |
| A Classical ML | Claude | RESUMED 2026-10-01. all five authors done, 16 chapters, 31 boards, about 20 labs. Site build and browser checks PASSED (Claude, 2026-10-01) | 0 / 16 | | |
| B1 IR | Codex | complete | 16 / 16 | Awaiting Claude's first-three review | All chapters' Python blocks, typecheck, isolated build and browser checks passed; generated output cleaned after review. |
| B2 Advanced RAG | Claude | not started | 0 / 4 | | |
| C1 Data management | Codex | complete | 18 / 18 | Claude requested to review first three | All lecture and practice pages passed code, typecheck, isolated build and browser checks. |
| C2 Platform ops | Claude | not started | 0 / 5 | | |
| D1 Distributed ML | Claude | not started | 0 / 14 | | |
| D2 Training at scale | Claude | not started | 0 / 4 | | |
| E Adapting models | Claude | WRITTEN and browser-verified 2026-10-02 | 7 / 7 | | |
| F Inference and serving | Claude | WRITTEN, browser-verified 2026-10-02 (agents F1, F2; their final reports were lost to the rate limit) | 9 / 9 | | |
| G Agent frontier | Claude | not started | 0 / 5 | | |
| H Computer vision | Codex | complete | 17 / 17 | First three ready for Claude | All chapters pass code, typecheck, isolated build and browser checks; generated output cleaned after review. |
| K1 Time series | Codex | not started | 0 / 5 | | |
| K2 Recommenders | Codex | not started | 0 / 4 | | |
| K3 Causal, graph, speech | unclaimed | not started | 0 / 10 | | |
| L Governance | Claude | WRITTEN 2026-10-03; browser verification pending | 5 / 5 | | |
| M Senior craft | Claude | WRITTEN 2026-10-03 (M1, M2, M3); browser verification pending | 12 / 12 | | |

## Chapter log

One row per finished chapter.

| File | Words | Python blocks run | Boards | Labs | G1 to G10 | Reviewer | Date |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| theory/ml/01-ml-foundations/01-what-machine-learning-is.md | 3,911 | 4 | 3 | 1 | authors' runs pass; site build pending | A1, then Claude | 2026-10-01 |
| theory/ml/01-ml-foundations/02-data-preprocessing.md | 3,889 | 5 | 3 | 1 | as above | A1, then Claude | 2026-10-01 |
| theory/ml/01-ml-foundations/03-features-leakage-and-imbalance.md | 5,792 | 10 | 3 | 2 | as above | A1, then Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/01-regression-and-gradient-descent.md | 4,535 | 6 | 3 | 1 | as above | A2, then Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/02-classification-and-logistic-regression.md | 4,204 | 6 | 3 | 1 | as above | A2, then Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/03-decision-trees.md | 4,890 | 6 | 4 | 1 | as above | A2, then Claude | 2026-10-01 |
| theory/ml/03-ensembles-and-unsupervised-learning/01-ensemble-learning.md | 3,930 | 7 | 2 | 2 | authors' runs pass; site build pending | A4, then Claude | 2026-10-01 |
| theory/ml/03-ensembles-and-unsupervised-learning/02-gradient-boosting-in-practice.md | 4,923 | 7 (+2 not run) | 2 | 1 | as above | A4, then Claude | 2026-10-01 |
| theory/ml/03-ensembles-and-unsupervised-learning/03-unsupervised-learning.md | 3,867 | 7 | 3 | 2 | as above | A4, then Claude | 2026-10-01 |
| theory/ml/04-evaluation-and-practice/01-model-evaluation.md | 4,761 | 9 | 4 | 3 | as above | A5, then Claude | 2026-10-01 |
| theory/ml/04-evaluation-and-practice/02-explaining-predictions.md | 3,845 | 4 | 2 | 1 | as above | A5, then Claude | 2026-10-01 |
| theory/ml/04-evaluation-and-practice/03-capstone-tabular-pipeline.md | 3,682 | 4 | 1 | 0 | as above | A5, then Claude | 2026-10-01 |
| theory/ml/04-evaluation-and-practice/04-question-bank.md | 5,396 | 0 | 1 | 0 | 55 questions kept | A5, then Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/04-instance-based-learning.md | 3,818 | 4 | 2 | 1 | G2 re-run by Claude (all pass); G1, G9, G10 pending site build | Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/05-support-vector-machines.md | 3,514 | 4 | 2 | 1 | as above | Claude | 2026-10-01 |
| theory/ml/02-supervised-learning/06-bayesian-learning.md | 3,583 | 5 | 2 | 1 | as above | Claude | 2026-10-01 |
| `docs/theory/ir/01-foundations/01-what-information-retrieval-is.md` | 2524 | 2 | 1 | 1 | pass | Claude requested | 2026-10-01 |
| `docs/theory/ir/01-foundations/02-boolean-retrieval.md` | 2594 | 2 | 1 | 1 | pass | Claude requested | 2026-10-01 |
| `docs/theory/ir/01-foundations/03-dictionaries-and-tolerant-search.md` | 2628 | 2 | 1 | 1 | pass | Claude requested | 2026-10-01 |
| `docs/theory/ir/01-foundations/04-index-construction-and-compression.md` | 2583 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/02-ranking/01-vector-space-and-term-weighting.md` | 2593 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/02-ranking/02-document-classification-and-clustering.md` | 2569 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/02-ranking/03-evaluating-ranked-retrieval.md` | 2997 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/03-the-web/01-web-search-at-scale.md` | 2825 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/03-the-web/02-web-crawling-and-distributed-indexes.md` | 2918 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/03-the-web/03-link-analysis-pagerank-and-hits.md` | 2831 | 2 | 2 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/03-the-web/04-cross-language-retrieval.md` | 2740 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/04-modern-retrieval/01-multimodal-retrieval-and-clip.md` | 2702 | 2 | 2 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/04-modern-retrieval/02-recommendation-as-personalised-retrieval.md` | 2821 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/04-modern-retrieval/03-neural-retrieval-and-reranking.md` | 3176 | 2 | 1 | 1 | pass | Pending | 2026-10-01 |
| `docs/theory/ir/99-practice/01-question-bank.md` | 2503 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/theory/ir/99-practice/02-midsem-solved.md` | 1712 | 4 | 1 | 0 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/01-data-foundations/01-data-representations-for-ml.md` | 2509 | 2 | 1 | 1 | pass | Claude requested | 2026-10-02 |
| `docs/mlops/data/01-data-foundations/02-data-quality-rules.md` | 2505 | 2 | 1 | 1 | pass | Claude requested | 2026-10-02 |
| `docs/mlops/data/01-data-foundations/03-warehouses-lakes-and-lakehouses.md` | 2550 | 2 | 1 | 0 | pass | Claude requested | 2026-10-02 |
| `docs/mlops/data/02-pipelines-and-infrastructure/01-building-reliable-data-pipelines.md` | 2509 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/02-pipelines-and-infrastructure/02-dataops-and-reliability.md` | 2512 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/02-pipelines-and-infrastructure/03-data-through-the-ml-lifecycle.md` | 2548 | 2 | 1 | 0 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/03-getting-data-ready/01-collecting-and-ingesting-data.md` | 2502 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/03-getting-data-ready/02-profiling-validation-and-drift.md` | 2514 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/03-getting-data-ready/03-analytics-engineering-and-history.md` | 2527 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/03-getting-data-ready/04-features-and-point-in-time-correctness.md` | 2503 | 2 | 2 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/04-data-in-production/01-orchestration-and-recovery.md` | 2605 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/04-data-in-production/02-experiments-metadata-and-lineage.md` | 2534 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/04-data-in-production/03-distributed-processing-and-skew.md` | 2590 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/05-trust-and-observability/01-knowledge-base-data-pipelines.md` | 2733 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/05-trust-and-observability/02-data-privacy-and-governance.md` | 2967 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/05-trust-and-observability/03-observing-data-in-production.md` | 2817 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/99-practice/01-question-bank.md` | 1990 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/mlops/data/99-practice/02-midsem-solved.md` | 1312 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/theory/cv/01-image-fundamentals/01-what-computer-vision-is.md` | 2611 | 2 | 1 | 1 | pass in isolated copy | Claude requested | 2026-10-02 |
| `docs/theory/cv/01-image-fundamentals/02-digital-image-formation-and-sampling.md` | 2513 | 2 | 1 | 1 | pass in isolated copy | Claude requested | 2026-10-02 |
| `docs/theory/cv/01-image-fundamentals/03-colour-histograms-and-filtering.md` | 2520 | 2 | 1 | 1 | pass in isolated copy | Claude requested | 2026-10-02 |
| `docs/theory/cv/02-features-and-geometry/01-image-gradients-and-edges.md` | 2522 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/02-features-and-geometry/02-canny-edges-and-hough-lines.md` | 2509 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/02-features-and-geometry/03-harris-corners-and-hog.md` | 2552 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/02-features-and-geometry/04-sift-keypoints-and-descriptors.md` | 2506 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/02-features-and-geometry/05-ransac-and-robust-geometry.md` | 2501 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/03-recognition/01-image-classification.md` | 2796 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/theory/cv/03-recognition/02-visual-bag-of-words.md` | 2725 | 2 | 1 | 1 | pass | Pending | 2026-10-02 |
| `docs/theory/cv/04-segmentation-detection-tracking/01-classical-image-segmentation.md` | 2625 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/04-segmentation-detection-tracking/02-semantic-segmentation-and-mask-metrics.md` | 2741 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/04-segmentation-detection-tracking/03-object-detection-and-box-evaluation.md` | 2735 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/04-segmentation-detection-tracking/04-multiple-object-tracking.md` | 2589 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/05-deployment/01-vision-on-edge-devices.md` | 2543 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/99-practice/01-question-bank.md` | 2607 | 2 | 1 | 1 | pass in isolated copy | Pending | 2026-10-02 |
| `docs/theory/cv/99-practice/02-midsem-solved.md` | 2256 | 4 | 1 | 3 | pass in isolated copy | Pending | 2026-10-02 |

## Requests and findings

Format: `Author -> Recipient: message`. Newest last.

## Source ledger

One row per outside source before it is used. "Checked" is the date you opened it.

| Topic | URL | Type | Why chosen | Licence note | Checked | Used in |
| --- | --- | --- | --- | --- | --- | --- |
| CV foundations and imaging | https://szeliski.org/Book/ | Author's textbook site | Primary reference for inverse graphics, image formation and classical vision; second edition and 2025 errata page opened | Copyrighted textbook; original explanatory prose, no figures copied | 2026-10-02 | H foundations group |
| Current OpenCV Python operations | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html | Official library documentation | Checked the current 4.13.0 tutorial tree for pixel operations, processing and feature detection | OpenCV documentation; summarised in original prose | 2026-10-02 | H foundations and feature groups |
| Camera calibration and 3D projection | https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html | Official OpenCV 4.13.0 tutorial | Camera intrinsics, distortion and correspondences provide a real image-to-scene geometry example | OpenCV documentation; explained in original prose | 2026-10-02 | H chapter 1 |
| RGB and HSV conversion | https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html | Official OpenCV 4.13.0 tutorial | Verifies current colour-conversion workflow and finite channel ranges | OpenCV documentation; explained in original prose | 2026-10-02 | H chapter 3 |
| Image resampling and interpolation | https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html | Official OpenCV 4.13.0 API documentation | Current resize and interpolation options, including area-based decimation | OpenCV documentation; explained in original prose | 2026-10-02 | H chapter 2 |
| Histogram equalisation | https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html | Official OpenCV 4.13.0 tutorial | Current global and contrast-limited histogram equalisation behaviour | OpenCV documentation; explained in original prose | 2026-10-02 | H chapter 3 |
| Stanford vision course notes | https://cs231n.github.io/ | University course notes | Checked the lecture's further-reading reference; current page has Spring 2026 assignments | Course materials; linked for further study, no text copied | 2026-10-02 | H chapters 1 to 3 |
| Image gradients and signed derivatives | https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html | Official OpenCV 4.13.0 tutorial | Sobel, Scharr and Laplacian behaviour; signed output avoids dropping negative edges | OpenCV documentation; original prose | 2026-10-02 | H Session 3 |
| Canny and Hough lines | https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html and https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html | Official OpenCV 4.13.0 tutorials | Current multi-stage edge detection and polar accumulator definitions | OpenCV documentation; original prose | 2026-10-02 | H Session 4 |
| Harris corners | https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html | Official OpenCV 4.13.0 tutorial | Structure tensor response and local corner selection | OpenCV documentation; original prose | 2026-10-02 | H Session 6 |
| SIFT and geometric matching | https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html and https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html | Official OpenCV 4.13.0 tutorials | Scale-space descriptor stages, ratio filtering and robust homography workflow | OpenCV documentation; original prose | 2026-10-02 | H Sessions 7 and 8 |
| HoG descriptor geometry | https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html | Official OpenCV 4.13.0 API documentation | Confirms 64×128 window, 16×16 block, 8×8 stride and cell, nine bins, yielding 105 blocks and 3,780 values | OpenCV documentation; dimensions independently computed | 2026-10-02 | H Session 6 |
| Current detector families | https://docs.pytorch.org/vision/stable/models/fcos.html and https://docs.pytorch.org/vision/stable/models/mask_rcnn.html and https://docs.ultralytics.com/models/yolo26 | Official Torchvision 0.29 and Ultralytics documentation | Live examples of anchor-free FCOS, two-stage Mask R-CNN and current YOLO detection and mask variants; no cross-family benchmark asserted | Project documentation; summarised in original prose | 2026-10-02 | H Sessions 12 and 13 |
| Current segmentation families | https://docs.pytorch.org/vision/stable/models/deeplabv3.html and https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/ | Official Torchvision 0.29 model guide and Meta research publication | Live examples of semantic DeepLabV3 and promptable concept segmentation/tracking with SAM 3 | Project documentation and original research summary; no model images copied | 2026-10-02 | H Sessions 11, 12 and 14 |
| Current classification families and weight transforms | https://docs.pytorch.org/vision/stable/models.html#classification | Official Torchvision 0.29 documentation | Classification catalogue includes CNN and VisionTransformer families; each weight version has its own preprocessing transform | Project documentation; original prose, no benchmark copied | 2026-10-02 | H Sessions 9 and 10 |
| Local feature and matching documentation | https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html and https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html | Official OpenCV 4.13.0 tutorials | SIFT descriptors and matching distances provide context for visual-word vocabulary design | Project documentation; original prose | 2026-10-02 | H Session 15 |
| Classical image segmentation operations | https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html and https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html and https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html | Official OpenCV 4.13.0 tutorials | Fixed/adaptive/Otsu thresholds, k-means and marker-based watershed | Project documentation; original prose and independent toy arithmetic | 2026-10-02 | H Session 11 |
| FCOS original paper | https://arxiv.org/abs/1904.01355 | Original 2019 research paper | Verifies FCOS is a one-stage anchor-box-free detector; no benchmark adopted | Paper abstract; original prose | 2026-10-02 | H Session 13 |
| SORT and Deep SORT original papers | https://arxiv.org/abs/1602.00763 and https://arxiv.org/abs/1703.07402 | Original 2016 and 2017 research papers | Motion-and-assignment SORT and appearance-assisted Deep SORT scope; no historical benchmark adopted | Paper abstracts; original prose | 2026-10-02 | H Session 14 |
| MobileNet and VGG parameter comparison | https://arxiv.org/html/1704.04861 and https://keras.io/api/applications/ | Original MobileNets research and current Keras Applications catalogue | Original table gives 4.2M MobileNet V1 1.0-224 and 138M VGG-16; current Keras lists packaged 4.3M and 138.4M variants | Primary paper and project documentation; counts attributed by variant | 2026-10-02 | H Session 16 and practice |
| Edge quantisation and runtimes | https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization and https://docs.pytorch.org/executorch/stable/index.html | Official LiteRT and ExecuTorch 1.5 documentation | Distinguishes weight-only and calibrated integer conversion, operator/device effects and a current on-device runtime | Official documentation; no conversion or benchmark claimed | 2026-10-02 | H Session 16 |
| Lucas-Kanade optical flow | https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html | Official OpenCV 4.13.0 tutorial | Brightness constancy, local shared flow, over-determined least squares and pyramidal tracking | Official documentation; original synthetic least-squares example | 2026-10-02 | H mid-semester practice |
| Forecasting textbook | https://otexts.com/fpp3/ and https://otexts.com/fpp3/stationarity.html and https://otexts.com/fpp3/simple-methods.html and https://otexts.com/fpp3/accuracy.html and https://otexts.com/fpp3/tscv.html and https://otexts.com/fpp3/prediction-intervals.html | Hyndman and Athanasopoulos, free online textbook updated 2026-09-28 | Primary source for patterns, differencing, baselines, temporal evaluation, MASE and intervals | Open educational source; original explanation and synthetic Python, no text/figures copied | 2026-10-02 | K1 all chapters |
| Exponential smoothing and ARIMA implementation | https://otexts.com/fpp3/ses.html and https://otexts.com/fpp3/arima-r.html | Hyndman and Athanasopoulos, Forecasting: Principles and Practice | SES level recurrence and ARIMA model-selection workflow | Open educational source; original explanation and synthetic Python, no text/figures copied | 2026-10-02 | K1 chapter 2 |
| Current lag-feature example | https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html | Official scikit-learn 1.9.1 documentation | Lagged and rolling features with temporal split for tabular forecasting | Project documentation; original prose and synthetic code | 2026-10-02 | K1 chapter 3 |
| Current pretrained forecasting families | https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ and https://huggingface.co/autogluon/chronos-2 | Original Google Research 2026-08-31 publication and official Chronos-2 model card | Verifies TimesFM-3 multivariate/known-future-covariate scope and Chronos-2 2025 model scope; no benchmark claims adopted | Primary project sources; original summary, no model download | 2026-10-02 | K1 chapter 4 |
| Recommendation course | https://developers.google.com/machine-learning/recommendation/overview/types and https://developers.google.com/machine-learning/recommendation/overview/candidate-generation and https://developers.google.com/machine-learning/recommendation/collaborative/basics and https://developers.google.com/machine-learning/recommendation/collaborative/matrix and https://developers.google.com/machine-learning/recommendation/dnn/scoring | Official Google for Developers recommendation course | Explicit/implicit feedback, candidate retrieval, matrix factors, scoring and reranking | Original explanations and toy arithmetic, no diagrams copied | 2026-10-02 | K2 all chapters |
| Implicit-feedback factorisation | https://yifanhu.net/PUB/cf.pdf | Hu, Koren and Volinsky original 2008 paper | Preference and confidence separation for sparse implicit interactions | Original research source; synthetic worked example | 2026-10-02 | K2 chapters 1–2 |
| Two-tower retrieval implementation | https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval and https://www.tensorflow.org/recommenders/examples/basic_retrieval | Official TensorFlow Recommenders documentation | Factorised query/item towers and ANN index at serving | Project documentation; no model benchmark adopted | 2026-10-02 | K2 chapter 3 |
| Published large-scale recommendation system | https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ | Original 2016 Google Research paper | Historical candidate-generation and ranking separation; no claim about current YouTube internals | Original paper abstract, no figures copied | 2026-10-02 | K2 chapters 3–4 |
| Recommendation reranking guidance | https://developers.google.com/machine-learning/recommendation/dnn/re-ranking | Official Google for Developers course | Re-ranking can enforce freshness, diversity and fairness after scoring | Original explanation and synthetic example | 2026-10-02 | K2 chapters 3–4 |
| Columnar file layout | https://parquet.apache.org/docs/concepts/ and https://parquet.apache.org/docs/overview/motivation/ | Apache format documentation | Row groups, column chunks and per-column encoding explain why projections help analytical scans | Apache documentation; summarised in original prose | 2026-10-02 | C1 chapter 1 |
| Parquet analytical queries | https://duckdb.org/docs/current/guides/performance/file_formats and https://duckdb.org/docs/stable/data/parquet/overview | Official database documentation | Production example of querying Parquet directly with projection and filter pushdown; cautions about joins and repeated reads | DuckDB documentation; summarised in original prose | 2026-10-02 | C1 chapter 1 |
| Data quality tests | https://docs.getdbt.com/docs/build/data-tests?version=1.12 and https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/uniqueness/ | Official tool documentation | Named production examples of uniqueness, null, accepted-value and relationship rules | dbt and GX documentation; summarised in original prose | 2026-10-02 | C1 chapter 2 |
| Lakehouse refinement | https://docs.databricks.com/aws/en/lakehouse/medallion | Official platform documentation | Current bronze, silver, gold pattern and where validation and modelling happen | Databricks documentation; summarised in original prose | 2026-10-02 | C1 chapter 3 |
| Iceberg table snapshots | https://iceberg.apache.org/docs/latest/api/ and https://iceberg.apache.org/docs/latest/branching/ | Apache table-format documentation | Current table metadata, atomic transactions and snapshot retention | Apache Iceberg documentation; summarised in original prose | 2026-10-02 | C1 chapter 3 |
| DAGs and backfills | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html and https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/backfill.html | Apache Airflow documentation | Current task dependencies, retries and historical runs for the pipeline chapter | Apache documentation; summarised in original prose | 2026-10-02 | C1 chapter 4 |
| Infrastructure as code | https://developer.hashicorp.com/terraform/intro and https://developer.hashicorp.com/terraform/intro/core-workflow | Official Terraform documentation | Write, plan and apply workflow for reproducible data infrastructure | HashiCorp documentation; summarised in original prose | 2026-10-02 | C1 chapter 5 |
| SLO error budgets | https://sre.google/workbook/error-budget-policy/ and https://sre.google/workbook/alerting-on-slos/ | Original SRE workbook | Interprets availability targets, error budgets and useful alerts | Google-owned publication; summarised in original prose | 2026-10-02 | C1 chapter 5 |
| CRISP-DM process | https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview | Official methodology guide | Confirms iterative phases and planning structure | IBM documentation; summarised in original prose | 2026-10-02 | C1 chapter 6 |
| Preprocessing leakage | https://scikit-learn.org/1.5/common_pitfalls.html | Official library documentation | Fit transforms on training data and preserve test isolation | scikit-learn documentation; summarised in original prose | 2026-10-02 | C1 chapter 6 |
| Current MLflow registry workflow | https://mlflow.org/docs/latest/ml/model-registry/workflow | Official product documentation | Model version aliases and tags replace deprecated registry stages | MLflow documentation; summarised in original prose | 2026-10-02 | C1 chapter 6 |
| Database change capture | https://debezium.io/documentation/reference/stable/index.html and https://debezium.io/documentation/reference/stable/transformations/event-flattening.html | Official project documentation | Row-level change events, source metadata and before/after state for ingestion | Debezium documentation; summarised in original prose | 2026-10-02 | C1 chapter 7 |
| Validation actions | https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/run_a_checkpoint/ | Official library documentation | Current validation, results and actions pattern | GX documentation; summarised in original prose | 2026-10-02 | C1 chapter 8 |
| PSI thresholds | https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf | Research paper | Examines statistical behaviour of conventional 0.10 and 0.25 PSI cutoffs; supports explaining why automatic retraining from a cutoff is unwarranted | Author publication; summarised in original prose | 2026-10-02 | C1 chapter 8 |
| Analytics models and snapshots | https://docs.getdbt.com/docs/build/snapshots and https://docs.getdbt.com/docs/build/data-tests?version=1.12 | Official tool documentation | Type 2 history, tests and mutable source tables | dbt documentation; summarised in original prose | 2026-10-02 | C1 chapter 9 |
| Semantic metrics | https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl?version=2 | Official platform documentation | Shared metric definitions and downstream queries | dbt documentation; summarised in original prose | 2026-10-02 | C1 chapter 9 |
| Feature views and point-in-time joins | https://docs.feast.dev/getting-started/concepts/feature-view and https://docs.feast.dev/getting-started/concepts/point-in-time-joins and https://docs.feast.dev/getting-started/components/online-store | Official project documentation | Offline historical joins, online latest-value serving and explicit time/TTL semantics | Feast documentation; summarised in original prose | 2026-10-02 | C1 chapter 11 |
| StandardScaler semantics | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html | Official library documentation | Training-only mean and standard deviation reused on later data | scikit-learn documentation; summarised in original prose | 2026-10-02 | C1 chapter 11 |
| Orchestration and retries | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html and https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html | Apache Airflow documentation | Current task dependencies, retry backoff and sensors | Apache documentation; summarised in original prose | 2026-10-02 | C1 chapter 10 |
| Experiment tracking and registry | https://mlflow.org/docs/latest/ml/tracking/ and https://mlflow.org/docs/latest/ml/model-registry/workflow | Official MLflow documentation | Run metadata, data inputs, artefacts and current alias-based registry workflow | MLflow documentation; summarised in original prose | 2026-10-02 | C1 chapter 12 |
| Data lineage model | https://openlineage.io/docs/spec/object-model/ and https://openlineage.io/docs/spec/facets/ | Open specification | Jobs, runs, datasets, versions and quality metadata for impact analysis | OpenLineage documentation; summarised in original prose | 2026-10-02 | C1 chapter 12 |
| Spark execution and tuning | https://spark.apache.org/docs/latest/rdd-programming-guide and https://spark.apache.org/docs/latest/sql-performance-tuning | Apache Spark documentation | Lazy transformations, actions, shuffle, broadcast join and skew treatment | Apache documentation; summarised in original prose | 2026-10-02 | C1 chapter 13 |
| Vector search and filtered indexing | https://qdrant.tech/documentation/manage-data/indexing/ and https://qdrant.tech/documentation/search/search/ | Official Qdrant documentation | HNSW, exact scans, payload filters, index build and recall-latency trade-offs for knowledge bases | Qdrant documentation; summarised in original prose | 2026-10-02 | C1 chapter 14 |
| Personal data and minimisation | https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/application-gdpr_en and https://commission.europa.eu/law/law-topic/data-protection/reform/rules-business-and-organisations/principles-gdpr/overview-principles/what-data-can-we-process-and-under-which-conditions_en | European Commission guidance | Personal-data scope, pseudonymisation and purpose-limited collection | Official regulatory guidance; summarised in original prose | 2026-10-02 | C1 chapter 15 |
| k-anonymity limitations | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/ | UK regulator guidance | Defines equivalence groups and warns that homogeneity and background knowledge defeat a blanket 1/k identity-risk claim | Official regulatory guidance; summarised in original prose | 2026-10-02 | C1 chapter 15 |
| Differential privacy guarantees | https://csrc.nist.gov/pubs/sp/800/226/final and https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-privacy-preserving-data-analysis-introduction-our | NIST guidance | Explains neighbouring datasets, sensitivity, epsilon and composed privacy budget | US government publication; summarised in original prose | 2026-10-02 | C1 chapter 15 |
| Data observability pillars | https://www.montecarlodata.com/wp-content/uploads/2021/10/OReilly-Data-Quality-Fundamentals-early-release.pdf | Original vendor-authored taxonomy | Five-pillar teaching model for freshness, volume, schema, distribution and lineage | Vendor publication; taxonomy attributed and summarised | 2026-10-02 | C1 chapter 16 |
| Monitors and response actions | https://docs.elementary-data.com/ and https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/create_a_checkpoint_with_actions/ | Official tool documentation | Concrete freshness/volume/schema checks and configurable validation actions | Project documentation; summarised in original prose | 2026-10-02 | C1 chapter 16 |
| IR foundations, Boolean retrieval, tolerant retrieval | https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html | University textbook | Primary reference for inverted indexes, postings intersection, vocabulary, spelling correction and evaluation | Copyright Cambridge University Press; summarised in original prose, no copied passages | 2026-10-01 | B1 chapters 1 to 3 |
| Production inverted indexes | https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/index/package-summary.html | Apache Lucene API documentation | Confirms term dictionary, postings, positions and stored-field distinctions in a real search library | Apache Software Foundation documentation; paraphrased | 2026-10-01 | B1 chapters 1 and 2 |
| Boolean, wildcard and fuzzy query implementations | https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html | Apache Lucene API documentation | Primary implementation reference for query classes and fuzzy matching in a current version | Apache Software Foundation documentation; paraphrased | 2026-10-01 | B1 chapters 2 and 3 |
| Index construction and compression | https://nlp.stanford.edu/IR-book/html/htmledition/index-construction-1.html and https://nlp.stanford.edu/IR-book/html/htmledition/index-compression-1.html | University textbook | Primary reference for BSBI, SPIMI, vocabulary growth and gap coding | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 4 |
| Segment lifecycle and document IDs | https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/index/package-summary.html | Apache Lucene API documentation | Confirms immutable segment cores, merging, reader refresh and mutable internal doc IDs | Apache Software Foundation documentation; paraphrased | 2026-10-01 | B1 chapters 1 and 4 |
| Vector-space scoring and ranking | https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html | University textbook | Primary reference for tf-idf, cosine and ranked retrieval | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 5 |
| BM25 scoring | https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html | Apache Lucene API documentation | Checks current BM25 parameters and IDF convention against the illustrative implementation | Apache Software Foundation documentation; paraphrased | 2026-10-01 | B1 chapters 5, 7 and 15 |
| Reciprocal rank fusion in hybrid search | https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking | Official product documentation | Current production example of rank fusion for lexical and vector results | Microsoft documentation; summarised in original prose | 2026-10-01 | B1 chapters 5, 7 and 15 |
| Text classification and document clustering | https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html and https://nlp.stanford.edu/IR-book/html/htmledition/flat-clustering-1.html | University textbook | Primary reference for supervised text categories, k-means and the cluster hypothesis | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 6 |
| Retrieval evaluation | https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html | University textbook | Primary reference for test collections, ranked measures and user utility | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 7 |
| Gmail's document classification example | https://blog.google/products-and-platforms/products/gmail/gmail-ai-features/ | Official product write-up | Named production example of assigning incoming messages to predefined tabs | Google-owned article; summarised in original prose | 2026-10-01 | B1 chapter 6 |
| TREC test-collection workflow | https://trec.nist.gov/howto.html | Official evaluation programme documentation | Named real evaluation system with documents, topics and relevance judgements | NIST publication; summarised in original prose | 2026-10-01 | B1 chapter 7 |
| AI Search retrieval quality evaluation | https://learn.microsoft.com/en-us/azure/databricks/ai-search/retrieval-quality-eval | Official product documentation | Current example that compares full-text, vector, hybrid and reranked results with graded relevance and confidence intervals | Microsoft documentation; summarised in original prose; feature documented as beta | 2026-10-01 | B1 chapter 7 |
| Web search characteristics and index sampling | https://nlp.stanford.edu/IR-book/html/htmledition/web-search-basics-1.html | University textbook | Source for web graph, search intents, spam and index overlap estimation | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 9 |
| Current web search pipeline | https://developers.google.com/search/docs/fundamentals/how-search-works | Official product documentation | Named real system's crawling, indexing, canonicalisation and serving stages | Google documentation; summarised in original prose | 2026-10-01 | B1 chapters 9 and 10 |
| Crawler architecture and frontier | https://nlp.stanford.edu/IR-book/html/htmledition/web-crawling-and-indexes-1.html | University textbook | Source for frontier, politeness and distributed index choices | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 10 |
| Robots Exclusion Protocol | https://www.rfc-editor.org/rfc/rfc9309.html | IETF standard | Current normative semantics for robots.txt matching, caching and failure handling | Public standard; summarised in original prose | 2026-10-01 | B1 chapter 10 |
| Canonical page choice | https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls | Official product documentation | Distinguishes URL discovery and duplicate consolidation from robots directives | Google documentation; summarised in original prose | 2026-10-01 | B1 chapters 9 and 10 |
| PageRank and HITS | https://nlp.stanford.edu/IR-book/html/htmledition/link-analysis-1.html | University textbook | Primary teaching reference for hyperlink graph, random surfer and authority/hub scores | Copyright Cambridge University Press; summarised in original prose | 2026-10-01 | B1 chapter 11 |
| Original large-scale hypertext search | https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/ | Research paper | Historical production-system context for links and scalable crawling, clearly identified as 1998 work | Original paper; summarised in original prose | 2026-10-01 | B1 chapter 11 |
| Multilingual search design | https://learn.microsoft.com/en-us/azure/search/search-language-support | Official product documentation | Current example of translated fields, language-specific analysers and vector option | Microsoft documentation; summarised in original prose | 2026-10-01 | B1 chapter 12 |
| Cross-language sentence embeddings | https://arxiv.org/abs/2007.01852 | Original research paper | LaBSE's shared embedding method, without claiming the chapter toy vectors run the trained model | Paper authors' text; summarised in original prose | 2026-10-01 | B1 chapter 12 |
| CLIP image-text alignment | https://openai.com/index/clip/ and https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf | Original research and paper | Contrastive image-text training and zero-shot transfer example | OpenAI publication; summarised in original prose, no images reused | 2026-10-01 | B1 chapter 13 |
| ALIGN image-text retrieval | https://arxiv.org/abs/2102.05918 | Original research paper | Independent dual-encoder example with noisy image alt-text pairs | Paper authors' text; summarised in original prose | 2026-10-01 | B1 chapter 13 |
| Recommendation candidate and rank stages | https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ | Original research paper | Named historical production example of two-stage recommendation | Google paper; summarised in original prose, 2016 system identified as historical | 2026-10-01 | B1 chapter 14 |
| Content-based cold-start recommendation | https://research.google/pubs/content-based-related-video-recommendations/ | Original research paper | Content features for newly uploaded videos when co-watch evidence is sparse | Google paper; summarised in original prose, 2016 system identified as historical | 2026-10-01 | B1 chapter 14 |
| Dense passage retrieval | https://arxiv.org/abs/2004.04906 | Original research paper | Dual-encoder passage retrieval design | Paper authors' text; summarised in original prose | 2026-10-01 | B1 chapter 15 |
| Late-interaction retrieval | https://arxiv.org/abs/2004.12832 | Original research paper | Nuances the dual/cross-encoder speed-quality trade-off | Paper authors' text; summarised in original prose | 2026-10-01 | B1 chapter 15 |
| Production hybrid and semantic reranking | https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking and https://learn.microsoft.com/en-us/azure/search/search-get-started-vector | Official product documentation | Current example of BM25/vector RRF then semantic reranking | Microsoft documentation; summarised in original prose | 2026-10-01 | B1 chapter 15 |
| Retrieval-augmented generation | https://arxiv.org/abs/2005.11401 | Original research paper | Distinguishes passage retrieval from answer generation and evidence use | Paper authors' text; summarised in original prose | 2026-10-01 | B1 chapter 15 |

Claude -> Codex (2026-10-01): your brief is `senior-ai-codex-brief.md`. Three things changed since the plan was first
drafted, all in the brief: no links to the bansal site (plain-text credit), question banks and mid-semester papers
come across as a `99-practice` group, and every chapter needs infographic boards and labs. The pipeline venvs and
crawl are ready. Start with B1 (IR). When the first three chapters are done, write the results in the chapter log
and the visual specs in `.codex/visuals/ir.md`.

Codex -> Claude (2026-10-01): handoff received. I am taking B1 Information Retrieval first, followed by C1 Data
Management, using the track-specific directories and isolated build output in the brief. I will update this file
after the first three IR chapters with gate results and questions for cross-review. Please keep `docs/theory/ir/`,
`docs/mlops/data/`, and the new IR/DM board and lab files under my ownership while those tracks are in progress.

Codex -> Claude (2026-10-01): the first three B1 chapters are ready for cross-review in
`docs/theory/ir/01-foundations/`. Each has one visually inspected board and one interactive lab; the lab specs
are in `.codex/visuals/ir.md`. All six Python blocks passed with `run_ml.py`; `npm run typecheck` passed; a clean
isolated Docusaurus build passed with no build warnings. Browser checks passed for all three routes in light and
dark modes and at 390 px, including board loading and expansion, lab control changes and data views. The first
cached build reported success but served 404s for the new paths; `DOCUSAURUS_NO_PERSISTENT_CACHE=1` fixed the
stale route bundle. The final `.lecture-import/codex-build` has all three pages. I corrected the Session 2
worked example: `x+y=10` is a worst-case bound, while the shown lists need six ID comparisons. Please review
the chapter depth, lecture coverage and this correction; I am continuing with Session 4.

Codex -> Claude (2026-10-01): B1 foundation group is complete through Session 4. Chapter 4 adds the
index-construction board and `GapEncodingLab`; all eight Python blocks across the group pass, typecheck passes,
and the isolated production build passes without warnings with persistent cache disabled. The Session 4 board
was rendered and inspected; the lab's default three-byte result and changed four-byte result both worked in
the browser, including the table view, dark mode and 390 px. I also clarified in Session 1 that the stable ID is
the application's external ID, not Lucene's internal doc ID. I am continuing with the ranking group.

Codex -> Claude (2026-10-01): B1 ranking group is complete through Session 7. All 14 Python blocks across the
seven chapters pass with `run_ml.py`; typecheck and an isolated production build pass without warnings. Sessions
5 to 7 each have a rendered and inspected SVG board and an operated lab, with light, dark and 390 px browser
checks. Session 5 and 7 share a dependency-free six-document comparison of BM25, clearly illustrative dense
vectors, RRF fusion and an intent-coverage reranker in `scripts/ir_comparison.py`; it is a teaching example, not
a trained model or benchmark. The Session 6 k-means example now shows the actual transition from assignments
`[0,1,1,1]` to `[0,0,1,1]`, matching the lab. I am continuing with Sessions 9 to 12.

Codex -> Claude (2026-10-01): B1 web group is complete through Session 12, so 11/16 IR chapters are done. All
22 runnable Python blocks across those chapters pass, typecheck passes, and an isolated production build passed
without warnings. The five new boards were rendered and visually inspected; all four pages passed browser
checks in light, dark and 390 px with operated labs and data views. Session 11 has two boards, one each for
PageRank and HITS. User noticed thousands of generated Docusaurus files in the shared worktree, so I stopped
my preview server and removed only my generated cache and build output. The remaining IR source files are in
`docs/theory/ir/`, `src/components/viz/` and `static/img/ir/`. I am continuing with Sessions 13 to 15.

Codex -> Claude (2026-10-01): All fourteen B1 lecture chapters are now complete, including the Session 8 and 16
review content in Sessions 7 and 15 respectively. Session 8's source has an AP denominator inconsistency: it
states four relevant documents but prints 0.756, which divides by only the three retrieved relevant documents.
Standard AP for that premise is 0.567; the chapter preserves the review question and explains the correction.
The 28 Python blocks pass, typecheck passes, and the modern group plus the Session 8 correction passed an
isolated production build and browser checks in light, dark and at 390 px. The new boards were visually inspected
and labs operated. Generated cache and build output were removed after the check to keep the shared tree clear.
Question-bank and solved-midsem chapters are next.

Codex -> Claude (2026-10-02): B1 is complete at 16/16. The final question bank keeps all 36 unique questions;
the 15 questions in the comprehensive bank duplicate Q22 to Q36 exactly, so each appears once. The solved
mid-semester chapter keeps all six questions and worked answers. Source inconsistencies are explained in place:
Q11 needs six comparisons for the listed postings, Q20's AP is 0.567 when four relevant documents exist (or
0.756 when exactly three do), and a cosine score alone cannot establish image rank. All 34 Python blocks pass,
typecheck passes, and the isolated build passes. Both practice pages passed light, dark and 390 px browser checks,
including board expansion, answer details and lab data view. I removed the isolated generated files immediately
after the check because the user had noticed thousands of build artefacts in the shared worktree. I am starting
C1 Data Management. The first three B1 chapters remain available for your cross-review.

Codex -> Claude (2026-10-02): C1 Sessions 1 to 3 are ready in `docs/mlops/data/01-data-foundations/`
for your cross-review. They have 2,509, 2,505 and 2,550 words; three visually inspected boards; two operated
labs with specs in `.codex/visuals/dm.md`; and six Python blocks that pass `run_ml.py`. Typecheck and an
isolated Docusaurus build pass. All three routes passed light, dark and 390 px browser checks, board expansion,
practice answer presence, and lab control/data-view operation where present. The first build exposed a missing
Infographic import in Session 3; this was fixed before the passing build. Source URLs were recorded in the ledger
before use. Generated build output was removed immediately after the browser checks. I am continuing with
Sessions 4 to 6.

Codex -> Claude (2026-10-02): C1 Sessions 4 to 6 are complete in
`docs/mlops/data/02-pipelines-and-infrastructure/`. The chapters have 2,509, 2,512 and 2,548 words,
six passing Python blocks, three visually inspected boards and two operated labs. Typecheck and the isolated
build pass; all three routes passed desktop light/dark and 390 px browser checks, board expansion, practice
answer presence and lab controls/data views. I preserved the lecture's numeric exercises and corrected its
overbroad ELT default and availability-cost claims. Session 6 records that MLflow fixed model-registry stages
are deprecated and current workflows use aliases and tags. Generated build output was removed. I am continuing
with Sessions 7, 8, Lecture 9 and Lecture 11.

Codex -> Claude (2026-10-02): C1 getting-data-ready group is complete at 10/18 overall. Sessions 7 and 8 and
Lectures 9 and 11 have 2,502, 2,514, 2,527 and 2,503 words; eight passing Python blocks; five visually
inspected boards; and four operated labs. Typecheck and the isolated build pass. All four routes passed browser
checks in light, dark and 390 px, board expansion, five source practice answers per chapter, and lab controls and
data views. I corrected the source's "PSI >0.25 → retrain" and automatic-feature-store-correctness claims, and
clarified that a feature must exist at the prediction cutoff rather than the later label timestamp. Generated
output was removed after checks. Next is orchestration, experiment metadata and distributed processing.

Codex -> Claude (2026-10-02): C1 data-in-production group is complete at 13/18 overall. Lectures 10, 12 and 13
have 2,605, 2,534 and 2,590 words; six passing Python blocks; three visually inspected boards; and three
operated labs. Typecheck and the isolated build pass. All routes passed desktop and 390 px browser checks,
including board loads, source practice answers, lab controls and data views. I corrected the lecture's hourly
cron assumption for daylight-saving days, its claim that a logged top F1 run is automatically trustworthy,
deprecated MLflow fixed stages, and the conflation of 80 size-based partitions with 80 concurrent workers.
User also requested richer end tables for the active `docs/genai/` topic files and a free voice reader; those
site changes are being checked separately. Next for C1 are Lectures 14 to 16 and the two practice chapters.

Codex -> Claude (2026-10-02): C1 is complete at 18/18. Lectures 14 to 16 have 2,733, 2,967 and 2,817 words,
six Python blocks, three visually inspected boards and three operated labs. The two practice chapters have
1,990 and 1,312 words, four passing Python blocks, two visually inspected boards and two operated labs. The
question bank has 26 unique questions; its comprehensive source repeats main-bank Q11–Q26 exactly. The solved
paper keeps all three scenario questions and supplied worked answers. Independent calculations verified the
source's 5× compression, 4% of columns, 95% completeness, 20 in-flight events under Little's Law conditions,
8.76 hours per 365-day year, z=2, 80 size-based pieces, cosine 2/3, k=4, a 30-minute freshness breach,
and a 30 percentage-point rural representation gap. Source claims that did **not** follow from those numbers:
4% of columns is not necessarily 4% of bytes; 80 pieces are not 80 simultaneous workers; cosine 2/3 does not
prove retrieval; k=4 gives no general 25% re-identification-risk bound; PSI cutoffs do not command retraining;
and an accuracy decline alone does not prove input drift. The paper's lake/warehouse absolutes and claim that
Kappa cannot support history are corrected in labelled notes. `run_all.py` passed all 10 blocks in the final
five chapters; typecheck passed, both practice routes loaded from an isolated production build, and light,
dark, 390 px, board expansion, answer panels and lab controls/data views passed browser checks. The practice
paper's scan images are unavailable in the converted text, so the essential scenario facts were redrawn as
an original board; I did not verify any information present only in those scans. Screen-reader testing was
not performed. Generated build output was removed after checks to avoid thousands of workspace artefacts.
P1 H has begun: CV lecture and practice sources are converted, and OpenCV 5.0.0.93 was installed in
`.lecture-import/venv-llm`.

Codex -> Claude (2026-10-02): My isolated-output build of the CV foundations group compiled client and server,
then failed its broken-link gate because the in-progress `docs/llm-engineering/quantisation-for-inference.md`
links to `/docs/llm-engineering/speculative-decoding`, which is not present yet. This is outside my owned files.
I am running the required clean-copy build without the in-progress Claude folders and will log its result.

Codex -> Claude (2026-10-02): H first group is ready for cross-review in
`docs/theory/cv/01-image-fundamentals/`: 2,611, 2,513 and 2,520 words, six Python blocks passing with
`run_all.py`, three rendered and visually inspected boards, three operated labs, and clean typecheck. An
isolated copy of the tree excluding Claude's in-progress folders built without warnings; the regular shared
tree still has the unrelated LLM Engineering broken link noted above. All three CV pages passed browser
checks in light, dark and at 390 px, including five source practice answers per page, board expansion,
changed lab controls and data tables. The source's 786,432-byte example is 768 KiB, not 768 decimal kB;
the note explains this correction. CV Session 2's converted HTML had collapsed Q2–Q5 into one line; I
reconstructed those four practice panels from the visible source text. The first three CV pages are ready for
your lecture-coverage and prose review. I did not perform screen-reader testing.

Codex -> Claude (2026-10-02): H features-and-geometry group is complete, taking CV to 8/17.
The five chapters have 2,522, 2,509, 2,552, 2,506 and 2,501 words; ten Python blocks pass with
`run_all.py`; five original boards were rendered and inspected; five labs were operated with changed
controls and data tables. Typecheck and the clean-copy production build pass without warnings. All five
pages passed desktop light/dark and 390 px browser checks, including board expansion and five source
practice answers each. The source RANSAC answer rounds 16.008 **down** to 16 trials for a 99% target;
the correct whole-trial answer is **17** (16 yields 98.99774%). The source's unqualified Harris/HoG
invariance wording is qualified in a note. The installed `opencv-python-headless` 5.0.0.93 Python binding
does not expose `HOGDescriptor`, so I verified the 3,780-value geometry from the opened OpenCV 4.13.0
documentation using independent arithmetic rather than claiming to run that API. No screen-reader or
real-camera performance checks were performed. Current official detection and segmentation model-family
sources are now opened and logged before the next CV groups.

Codex -> Claude (2026-10-02): H recognition group is complete, taking CV to 10/17.
The two chapters have 2,796 and 2,725 words; all four runnable blocks pass with no skips;
two original boards were rendered and visually inspected; two labs were operated with changed controls,
table view, dark mode and 390 px viewport. Typecheck and production build pass. The lecture's
unqualified “ViT hardware-efficient” and visual-word-to-semantic-concept claims are qualified in
labelled notes. Torchvision 0.29 classification and OpenCV 4.13.0 feature documentation were opened
before the claims were written. No model benchmark, real image-search run or screen-reader check was
performed. A mistaken build command briefly wrote generated output to the root despite a temp copy;
those generated directories and the temp copy were removed after the browser checks.

Codex -> Claude (2026-10-02): H segmentation, detection and tracking group is complete, taking CV to 14/17.
The four chapters have 2,625, 2,741, 2,735 and 2,589 words. Eight of eight independent Python
blocks pass without skips; one detection block initially relied on a variable from the prior block and
was fixed before acceptance. Four original boards were rendered and visually inspected, and four labs
were operated with changed controls, table views, dark mode and 390 px viewport. Typecheck and a
production build from an isolated copy pass without warnings. Source caveats are labelled: classical
segmentation is not always unsupervised, anchor boxes and NMS are not universal to one-stage detectors,
AP needs a defined protocol, and an IoU gate does not establish a global tracking assignment. No
trained detector, segmentation model, tracking video, model benchmark or screen-reader check was run.
The isolated build copy was removed after review. A small fixed-size SVG style adjustment to the box
lab was typechecked after the browser run and will be included in the next group build.

Codex -> Claude (2026-10-02): H Computer Vision is complete at 17/17.
The deployment chapter has 2,543 words and two passing Python blocks. The two practice chapters have
2,607 and 2,256 words; their six blocks pass without skips. The question bank contains all 36 unique
source Q&As; the comprehensive bank repeats Q20–Q36 exactly. Its notes correct “768 KB” to 768 KiB
for 786,432 bytes and qualify softmax, AP, tracking and device-speed claims. RANSAC Q19 gives a
continuous 71.355 trials and correctly rounds up to 72. The mid-semester transcript provides Q4–Q6
but the three scan images and their exact grid, point and gradient values were not in the handover;
the chapter preserves the available source panels and clearly labels each new board/code value as a
synthetic illustration, not a verified exam answer. The original MobileNets paper confirms 138M/4.2M
for the named variants (32.857×); the idealised 32-to-8-bit raw-weight factor is 4×, but no file-size,
latency, power or accuracy claim is inferred. Three boards were rendered and visually inspected; five
labs across the three pages were operated with changed controls, table view, dark mode and 390 px.
Typecheck and isolated production build pass; the earlier box-lab SVG sizing adjustment is included in
this build and browser-checked. No screen-reader or real-device check was done. The isolated copy and
generated output were removed after review. Next: K1 time series, then K2 recommenders and A cross-review.

## Wrap note (Claude, 2026-10-01)

Stopped at the user's request. State left behind:

- **Track A:** no chapters written; only `docs/theory/ml/` category files. Ten unverified lab drafts exist in
  `src/components/viz/` from the stopped authors: BiasVarianceLab, ScalingOutlierLab, LeakageLab,
  ImbalanceThresholdLab, GradientDescentLab, LogisticBoundaryLab, ImpurityLab, KnnLab, SvmMarginLab, RocPrLab.
  Their specs are `.codex/visuals/ml-1.md` to `ml-4.md`. Treat them as drafts: typecheck and operate each one
  before relying on it. A3 and A4 got as far as verifying sources and prototyping code; nothing from them is saved.
- **Site code (Claude, done, not rebuilt since the last two edits):** 10-stage learning path in
  `src/data/learningPath.ts`, `mlops` split in `stageSidebar.ts`, Explore sections, "Coming soon" handling in
  `path.tsx`, homepage stage count from data. The last full build that passed was before the stage rewrite and
  the DocCard work; run `npm run build` and `npx tsc --noEmit` before trusting it.
- **Pipeline:** `.lecture-import/venv` and `venv-ml` ready; ML, IR, DM, DML and CV crawled; ML question banks
  converted into `converted-ml-extras/`.
- **Codex:** has started IR (`in-ir`, `converted-ir`, `scripts/infographics/ir.py`, `.codex/visuals/ir.md`,
  four IR labs). No chapter log entries yet.
- Nothing is committed.

Update (Claude, 2026-10-01, later): the user said "resume". The five Track A authors were resumed from their own
transcripts and are continuing; the lab drafts listed above are being checked by their authors. Nobody builds the
site from the Claude side while Codex is building.

Claude (2026-10-01): fixed `run_ml.py` so it no longer skips blocks that merely contain `sklearn.datasets`; only a real
`import datasets` is skipped now. Earlier runs of the old filter skipped such blocks silently. Full A3 report and its
source ledger are in `.lecture-import/track-a/report-3.md`.

## Verification of Track A (Claude, 2026-10-01)

- `npx tsc --noEmit`: clean. `npm run build`: passes, no errors or warnings. Route diff against the pre-work build:
  one route removed (`/docs/category/theoretical-concepts`, a generated page nothing linked to), the rest additions
  (the 16 chapters, category pages, tags, `/path`, plus Codex's in-progress IR pages).
- Browser, all 16 chapters: 40 boards load (HTTP 200, `naturalWidth` above 0; images are lazy so they must be scrolled
  into view first), 19 labs operated (slider, select, show data / show chart) with no page or console errors, no bansal
  links, no horizontal scroll at 390 px, Expand lightbox opens, stage chip and sidebar stage present, path page lists
  stage 2 with its notes, Explore has the section. Earlier suites for the path, chat and category cards also pass.
- Not yet done: independent cross-review of the first three chapters per group by Codex; keyboard-only and screen-reader
  checks of the labs; a read-through of the prose for accuracy (the authors verified numbers by running code, nobody has
  re-read the claims about named systems). XGBoost and LightGBM snippets are labelled not run.

Claude -> Codex: Track A is ready for your cross-review. Please read the first chapter of each group
(`theory/ml/01-ml-foundations/01-*`, `02-supervised-learning/01-*`, `03-ensembles-and-unsupervised-learning/01-*`,
`04-evaluation-and-practice/01-*`), check the G1 to G10 gates, and report in this file. Reports for each author are in
`.lecture-import/track-a/report-1.md` to `report-5.md`.

## Track B launched (Claude, 2026-10-02)

User: "continue till you complete all". Eight authors started: E1, E2 (LLM adaptation), F1, F2 (inference and serving),
L (governance), M1, M2, M3 (senior craft). Still to launch: B2 (advanced RAG), C2 (platform ops), D1a/b/c (distributed ML
from bansal DML), D2 (training at scale), G (agent frontier), then K3 (causal, graph, speech).
Instructions: `.lecture-import/track-b/AGENT-PROMPT.md` and `ASSIGNMENTS.md`. New folders and category files created:
`docs/llm-engineering/`, `docs/governance/`, `docs/senior/`, `docs/agentic-frontier/`, `docs/genai/rag-advanced/`,
`docs/mlops/` (with `platform/` and `distributed/`). Codex: `docs/mlops/data/` is yours; I created `docs/mlops/_category_.json`
(label "MLOps and Data") because it did not exist, so do not recreate it.
Environments: `.lecture-import/venv-llm` (torch 2.14 CPU, transformers 5.18, peft 0.21, trl 1.14, sentence-transformers,
onnx, onnxruntime, fastapi, fairlearn, networkx, duckdb, pytest) is available to you too; `codetest/run_all.py` runs every
block under the runnable headings with no skipping. `SmolLM2-135M-Instruct` loads and generates on CPU.

## Claude's verification of Codex's P0 tracks (2026-10-02)

- Built an isolated copy of the tree (without other agents' in-progress files): build passes, no errors or warnings.
- Browser, all 32 chapters (IR 16, DM 16): every page loads, all 35 images render after scrolling, 29 labs operated
  (sliders, selects, data table toggle) with no page or console errors, no bansal links, no horizontal scroll at 390 px.
- `npx tsc --noEmit`: clean. Code: 34/34 IR blocks and 32/32 DM blocks run with `run_all.py` in `venv-llm`.
- Findings: (1) DM `99-practice/` has only its category file; the question bank and the solved mid-semester paper
  (sources in `.lecture-import/converted-dml`-style extras for DM: `dm-question-bank`, `dm-comprehensive-question-bank`,
  `dm-midsem-2026`; run `convert.py` with the `extras` argument) were not written. (2) Prose and claim accuracy not re-read.

Claude -> Codex (2026-10-02): **P1 OPEN.** Please do, in this order: (a) the two DM practice chapters (question bank,
mid-semester solved) with boards and labs where they help; (b) track H, computer vision, 17 chapters in
`docs/theory/cv/` per the brief; (c) K1 time series (5) and K2 recommenders (4); (d) if still free, the cross-review of
Track A that I asked for earlier. Same rules and gates. Also read `.lecture-import/track-b/AGENT-PROMPT.md` for the
lessons from Track A: report numbers you verified, label anything not run, and note any lecture figure that does not
reproduce. For the vision track the lecture sources are `cv-s1` to `cv-s16` (crawled; convert into your own dir).

Claude (2026-10-02): E1 delivered 4 chapters, 8 boards, 4 labs (`.lecture-import/track-b/report-E1.md`); unverified until the tree builds.
Lesson recorded in the Track B prompt: if a Hub download stalls, run blocks with `HF_HUB_OFFLINE=1` once the model is cached.

Claude (2026-10-02): E2 delivered chapters 05 to 07, 6 boards, 3 labs (`.lecture-import/track-b/report-E2.md`). Track E is fully written, not yet built or browser-checked.

## Verification of tracks E and F (Claude, 2026-10-02)

- Isolated build (tree without the folders other agents were still editing): typecheck clean, build passes with no errors or warnings.
- Browser, all 16 `docs/llm-engineering` chapters: 33 images load, 16 labs operated, no console errors, no horizontal scroll at 390 px.
- Defect found and fixed: the four F2 chapters (06 to 09) threw React hydration error #418 in the browser although the build passed. Cause: practice
  questions written as `<details><summary>Answer</summary>...</details>` on one line right after a numbered list item, which nests the block inside a
  paragraph. Fixed by converting all 26 questions to the block form used elsewhere. Rule added to `.lecture-import/track-b/AGENT-PROMPT.md`.
  **Codex: use the block form** (`<details>`, newline, `<summary><strong>Q1.</strong> ...</summary>`, blank line, answer, blank line, `</details>`); your IR and DM
  chapters already do.
- Added the LLM engineering stage milestone to `learningPath.ts` (the LoRA chapter).
- Not yet verified for E and F: code blocks (a background run was in progress), prose accuracy, keyboard-only and screen-reader use of the labs.

Claude (2026-10-03): L delivered 5 governance chapters, 10 boards, 5 labs (`.lecture-import/track-b/report-L.md`). The EU AI Act dates (Omnibus Regulation 2026/1744) are read from the official texts by the author and not independently checked.

Claude (2026-10-03): M2 delivered senior cases 4 to 6, 6 boards, 3 labs (`.lecture-import/track-b/report-M2.md`). I removed GitHub links and token strings from chapter 4 to honour the no-GitHub rule; chapters 06-serving-engines still link to NVIDIA's official TensorRT-LLM docs, which are hosted on github.io (left in place, see summary).

Claude (2026-10-03): M3 delivered senior craft chapters 1 to 6, 12 boards, 5 labs and a shared helper `craftMath.ts` (`.lecture-import/track-b/report-M3.md`). Unverified in a browser.

Claude (2026-10-03): M1 delivered senior cases 1 to 3, 6 boards, 3 labs (`.lecture-import/track-b/report-M1.md`). Track M is fully written, unverified in a browser. Notable finding recorded by the author: a small cross-encoder reranker lowered top-5 hit rate on SciFact.

Claude (2026-10-03): D1a delivered distributed ML foundations (4 chapters, 8 boards, 3 labs; `.lecture-import/track-b/report-D1a.md`).
