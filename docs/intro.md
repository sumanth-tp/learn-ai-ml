---
title: AI/ML learning roadmap
sidebar_position: 1
hide_table_of_contents: true
---

# AI/ML learning roadmap

# 1) Foundations (Weeks 1–6)

**Math you actually use**

- Linear algebra: vectors, matrices, SVD, eigendecomposition, norms.
- Probability & statistics: Bayes rule, distributions, expectation/variance,
  confidence intervals, hypothesis tests.
- Calculus for ML: gradients, Jacobians, chain rule, optimization intuition
  (convexity, learning rates).
- Optimization basics: gradient descent/variants, regularization (L1/L2), early
  stopping. **Why:** every model/debug task leans on these. **Deep dives:**
  Stanford CS229 page & notes (broad ML math context). ([CS229 Machine
  Learning][1])

**Programming**

- Python for data & ML: NumPy, pandas, matplotlib, scikit-learn;
  packaging/venv/poetry; testing & linting.
- Git/GitHub; basic shell; Jupyter/VS Code workflows. **LinkedIn roadmaps echo
  this as “non-negotiable.”** ([LinkedIn][2])

**Milestone project:** Titanic or Heart-Disease tabular pipeline (EDA → feature
engineering → model basket → metrics → cross-validation → error analysis →
report). **Reference post collections (LinkedIn):** curated
“basics→specialization” roadmaps. ([LinkedIn][3])

---

# 2) Core Machine Learning (Weeks 7–12)

**Supervised learning**

- Linear/logistic regression, SVMs, trees/ensembles (RF, XGBoost), k-NN;
  calibration; class imbalance; feature importance & SHAP. **Unsupervised**
- Clustering (k-means, DBSCAN), dimensionality reduction (PCA, t-SNE/UMAP).
  **Model selection**
- Bias–variance, regularization, cross-validation, leakage traps;
  production-grade metrics (AUROC, AUPRC, calibration). **Primary references:**
  CS229 (breadth), and several LinkedIn roadmaps emphasize a staged algorithmic
  path. ([CS229 Machine Learning][1])

**Milestone project:** End-to-end fraud/credit-risk or churn model with a
reproducible repo and a proper evaluation write-up.

---

# 3) Deep Learning (Weeks 13–20)

**Essentials**

- PyTorch workflow: tensors → autograd → modules → training loops → schedulers →
  mixed precision; CNNs, RNNs/GRUs/LSTMs, attention/Transformers (concepts).
- Regularization: dropout, weight decay; init; normalization; data augmentation;
  early stopping (a frequent Raschka tip). ([LinkedIn][4])

**Course-first approach:**

- fast.ai “Practical Deep Learning for Coders” (build then theory). ([Practical
  Deep Learning for Coders][5])
- Complement with Raschka’s PyTorch tutorials/posts for focused practice.
  ([LinkedIn][6])

**Milestone project:**

- Vision: defect detection or document OCR+classification.
- NLP: multi-label ticket routing. Ship a trained model + inference script +
  README.

---

# 4) MLOps & “Full-Stack ML” (Weeks 21–28)

**From notebooks to production**

- Data/versioning: DVC, LakeFS; experiment tracking (MLflow/W&B).
- Packaging & CI: tests, pre-commit, GitHub Actions; Docker.
- Serving: batch vs. online inference; FastAPI/gunicorn; vector DB basics for
  retrieval.
- Monitoring: drift, data quality, performance SLOs; feedback loops.
  **Authoritative, practical curricula:** Full Stack Deep Learning; community
  MLOps roadmap; curated GitHub roadmap. ([fullstackdeeplearning.com][7])

**Milestone project:** Package one of your earlier models as a **containerized
API** with offline evaluation, basic monitoring hooks, and a CI pipeline.

---

# 5) Generative AI & LLMs (Weeks 29–36)

**Foundations → applied**

- Tokenizers, embeddings, attention, pretraining vs. finetuning, prompting, RAG
  patterns, safety & evaluation.
- Build small: prompt engineering → eval harness → retrieval → guardrails →
  telemetry. **Short, focused courses and bootcamps:** DeepLearning.AI short
  courses and “Generative AI for Everyone”; FSDL LLM Bootcamp videos.
  ([DeepLearning.AI - Learning Platform][8])

**What practitioners emphasize on LinkedIn:** practical LLM strategy, MLOps for
GenAI, and project scoping (not just model training). Posts by Chip Huyen
repeatedly stress end-to-end thinking and deployment realities. ([LinkedIn][9])

**Milestone project:** A **domain RAG app** (e.g., compliance policy Q&A):
ingestion pipeline, chunking, retriever, reranker, eval set, latency & quality
dashboards.

---

# 6) Specializations (Weeks 37–44)

Choose one track and go deep:

- **NLP/LLMs:** instruction tuning, preference optimization, eval sets, tool use
  & agents.
- **Vision & Multimodal:** detection/segmentation, SAM, CLIP, diffusion.
- **Time Series / Forecasting:** feature windows, exogenous regressors,
  hierarchical forecasting, anomaly detection.
- **Reinforcement Learning (for ops or recsys):** policy gradient basics,
  offline/online RL safety. Use specialist lectures (fast.ai modules; CS229
  advanced topics) + targeted papers/repos. ([Practical Deep Learning for
  Coders][5])

---

# 7) Ethics, Privacy, and Responsible AI (ongoing)

- Data consent, bias/fairness testing, transparency docs, red-team evaluations
  for LLMs.
- Keep current with platform data-policy shifts (e.g., LinkedIn’s AI-training
  policy updates). ([The Verge][10])

---

# 8) Portfolio & Hiring Readiness (Weeks 45–52)

**Public proofs of work**

- 3–5 strong repos: clean READMEs, scripted pipelines, reproducible envs, and a
  short blog post per project.
- One production-style service (API + container + CI) and one GenAI app with
  evaluations. **Interview focus:** several LinkedIn “90-day interview roadmap”
  style posts outline consistent prep themes: math refreshers, classic
  algorithms, and project storytelling. ([LinkedIn][11])

---

## 90-Day Accelerated Plan (if you’re in a hurry)

- **Days 1–30:** Python + Math refresh (focused), classic ML with scikit-learn,
  2 tabular projects. ([LinkedIn][2])
- **Days 31–60:** PyTorch + fast.ai; ship 1 DL project (vision or NLP).
  ([Practical Deep Learning for Coders][5])
- **Days 61–90:** MLOps basics → containerize & serve; build a minimal RAG app
  and write an eval report. ([roadmap.sh][12])

---

## Curated “LinkedIn-Style” Reading List

These are representative, high-signal posts/threads you can reference (some may
require login to see content in full):

- AI/ML roadmap posts (basics → specialization) and checklists. ([LinkedIn][3])
- MLOps/GenAI strategy & what to learn now (Chip Huyen). ([LinkedIn][13])
- Hands-on DL & PyTorch tutorials (Sebastian Raschka). ([LinkedIn][6])
- Role-focused roadmaps (AI/ML engineer tracks). ([LinkedIn][14])

---

## Canonical Course & Reference Set (save/bookmark)

- **CS229 (Stanford)**—breadth + math. ([CS229 Machine Learning][1])
- **CS229 Notes PDF**—succinct, printable reference. ([CS229 Machine
  Learning][15])
- **fast.ai Practical DL**—build first, theory later. ([Practical Deep Learning
  for Coders][5])
- **Full Stack Deep Learning**—productization & MLOps.
  ([fullstackdeeplearning.com][7])
- **DeepLearning.AI courses & short courses**—targeted GenAI/LLM skills.
  ([DeepLearning.ai][16])
- **Roadmap.sh MLOps**—checklist perspective. ([roadmap.sh][12])

---

## How to use this roadmap

1. **Pick a cadence** (e.g., 8–10 hrs/week).
2. **Ship every 2–3 weeks:** a project with a README, metrics, and a short blog.
3. **Rotate focus:** ML → DL → MLOps → GenAI → specialization.
4. **Evaluate like a hiring team:** can someone run your code and see value in 5
   minutes?
5. **Stay current:** follow a handful of practitioners (Chip Huyen, Sebastian
   Raschka, Andrew Ng). ([LinkedIn][17])

[1]:
  https://cs229.stanford.edu/?utm_source=chatgpt.com
  "CS229: Machine Learning - Stanford University"
[2]:
  https://www.linkedin.com/posts/progressivethinker_ai-ml-roadmap-activity-7371083242780721152-HAH5?utm_source=chatgpt.com
  "How to Build Real-World ML Skills: A Complete Roadmap"
[3]:
  https://www.linkedin.com/posts/pranavkrishnavadhyar_machinelearning-deeplearning-ai-activity-7349395666428715008-yt_G?utm_source=chatgpt.com
  "AI/ML Roadmap: From Basics to Specialization"
[4]:
  https://www.linkedin.com/posts/sebastianraschka_deeplearning-ai-machinelearning-activity-7033799585852506112-E5eU?utm_source=chatgpt.com
  "Sebastian Raschka, PhD | 27 comments"
[5]:
  https://course.fast.ai/?utm_source=chatgpt.com
  "Practical Deep Learning for Coders - Practical Deep ... - Fast.ai"
[6]:
  https://www.linkedin.com/posts/sebastianraschka_pytorch-in-one-hour-from-tensors-to-training-activity-7346939468638994433-fPUK?utm_source=chatgpt.com
  "Learn PyTorch in 1 hour: Tensors to Multi-GPU Training"
[7]:
  https://fullstackdeeplearning.com/?utm_source=chatgpt.com
  "Full Stack Deep Learning"
[8]:
  https://learn.deeplearning.ai/?utm_source=chatgpt.com
  "DeepLearning.AI - Learning Platform"
[9]:
  https://www.linkedin.com/posts/chiphuyen_genai-llms-mlops-activity-7072595745282920448-Vj2Q?utm_source=chatgpt.com
  "Chip Huyen's Post - genai #llms #mlops"
[10]:
  https://www.theverge.com/2024/9/18/24248471/linkedin-ai-training-user-accounts-data-opt-in?utm_source=chatgpt.com
  "LinkedIn is training AI models on your data"
[11]:
  https://www.linkedin.com/posts/manishmazumder_if-you-spend-90-days-you-can-be-fully-prepared-activity-7352198106416926722-USDy?utm_source=chatgpt.com
  "AI/ML Engineer Interview Prep Roadmap in 90 Days"
[12]: https://roadmap.sh/mlops?utm_source=chatgpt.com "MLOps Roadmap"
[13]:
  https://www.linkedin.com/posts/chiphuyen_machinelearning-mlops-datascience-activity-7061181855726796800-eARV?utm_source=chatgpt.com
  "Chip Huyen - machinelearning #mlops #datascience"
[14]:
  https://www.linkedin.com/posts/jeanklee_roadmap-to-becoming-an-ai-or-ml-engineer-activity-7267580841642340352-P6jp?utm_source=chatgpt.com
  "Roadmap to Becoming an AI or ML Engineer! Build ..."
[15]:
  https://cs229.stanford.edu/main_notes.pdf?utm_source=chatgpt.com
  "CS229 Lecture Notes"
[16]: https://www.deeplearning.ai/courses/?utm_source=chatgpt.com "Courses"
[17]:
  https://www.linkedin.com/in/chiphuyen?utm_source=chatgpt.com
  "Chip Huyen - Building something new | AI x storytelling ..."
