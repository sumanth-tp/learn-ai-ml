---
title: W&B and MLflow Master Cheatsheet
sidebar_position: 16
---

# W&B and MLflow Master Cheatsheet

## Weights and Biases setup

| Method | Description | Code example |
|---|---|---|
| Install | Installs the W&B client. | `pip install wandb` |
| `wandb.login()` | Authenticates the local machine. | `import wandb`<br/>`wandb.login()` |
| `wandb.init()` | `wandb.init(project=None, entity=None, name=None, config=None)` starts a run. | `run = wandb.init(project="churn", name="xgb-baseline", config={"lr": 0.01, "epochs": 5})` |
| `wandb.log()` | Logs metrics over time. | `for epoch in range(5):`<br/>`    wandb.log({"epoch": epoch, "val_loss": 0.1 / (epoch + 1)})` |
| `wandb.finish()` | Ends the run cleanly. | `wandb.finish()` |
| Offline mode | Records locally and syncs later. | `WANDB_MODE=offline python train.py`<br/>`wandb sync wandb/offline-run-*` |

## W&B artifacts and sweeps

| Method | Description | Code example |
|---|---|---|
| `wandb.Artifact()` | Versioned files such as datasets, models, and reports. | `artifact = wandb.Artifact("model", type="model")`<br/>`artifact.add_file("model.pt")`<br/>`wandb.log_artifact(artifact)` |
| Use artifact | Downloads a tracked artifact into a local directory. | `artifact = run.use_artifact("model:latest")`<br/>`path = artifact.download()` |
| Watch model | Logs gradients and parameter histograms for PyTorch models. | `wandb.watch(model, log="gradients", log_freq=100)` |
| Sweep config | Defines hyperparameter search. | `sweep_config = {"method": "bayes", "metric": {"name": "val_loss", "goal": "minimize"}, "parameters": {"lr": {"values": [0.1, 0.01]}}}` |
| Sweep agent | Runs training jobs from a sweep. | `sweep_id = wandb.sweep(sweep_config, project="churn")`<br/>`wandb.agent(sweep_id, function=train)` |

## MLflow setup and runs

| Method | Description | Code example |
|---|---|---|
| Install | Installs MLflow. | `pip install mlflow` |
| `mlflow.set_tracking_uri()` | Points client to local or remote tracking server. | `mlflow.set_tracking_uri("http://localhost:5000")` |
| `mlflow.set_experiment()` | Selects or creates an experiment. | `mlflow.set_experiment("churn")` |
| `mlflow.start_run()` | Starts a tracking run. Use as context manager. | `with mlflow.start_run(run_name="baseline"):`<br/>`    mlflow.log_param("lr", 0.01)` |
| `mlflow.log_metric()` | Logs scalar metrics. | `mlflow.log_metric("val_accuracy", 0.92, step=1)` |
| `mlflow.log_params()` | Logs multiple parameters. | `mlflow.log_params({"lr": 0.01, "epochs": 10, "batch_size": 64})` |

## MLflow models and registry

| Method | Description | Code example |
|---|---|---|
| `mlflow.sklearn.log_model()` | Logs scikit-learn model with environment metadata. | `with mlflow.start_run():`<br/>`    mlflow.sklearn.log_model(model, artifact_path="model")` |
| `mlflow.pytorch.log_model()` | Logs PyTorch model. | `mlflow.pytorch.log_model(model, artifact_path="model")` |
| `mlflow.log_artifact()` | Logs a file artifact. | `mlflow.log_artifact("confusion_matrix.png")` |
| Register model | Adds a logged model to model registry. | `mlflow.register_model("runs:/RUN_ID/model", "ChurnClassifier")` |
| Load model | Loads model by URI. | `model = mlflow.pyfunc.load_model("models:/ChurnClassifier/Production")` |
| Serve model | Serves a logged model locally. | `mlflow models serve -m models:/ChurnClassifier/Production -p 5001` |

## Reproducibility and governance

| Method | Description | Code example |
|---|---|---|
| Log config | Store full config used by the run. | `mlflow.log_dict(config, "config.json")`<br/>`wandb.config.update(config)` |
| Log code version | Track git commit with each run. | `commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()`<br/>`mlflow.set_tag("git_commit", commit)` |
| Seed logging | Record seed and framework versions. | `mlflow.log_params({"seed": seed, "torch": torch.__version__})` |
| Dataset version | Track dataset artifact or checksum. | `wandb.log_artifact(dataset_artifact)`<br/>`mlflow.log_param("dataset_version", "s3://bucket/data/v4")` |
| Nested runs | Track parent run plus child trials. | `with mlflow.start_run(run_name="sweep"):`<br/>`    with mlflow.start_run(run_name="trial-1", nested=True):`<br/>`        mlflow.log_metric("loss", 0.2)` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Track training loop with W&B | Log train and validation metrics each epoch. | `run = wandb.init(project="mnist", config=config)`<br/>`for epoch in range(epochs):`<br/>`    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})` |
| Track training loop with MLflow | Use one context-managed run per experiment. | `with mlflow.start_run():`<br/>`    mlflow.log_params(config)`<br/>`    mlflow.log_metric("val_loss", val_loss)` |
| Log confusion matrix | Store evaluation artifacts. | `fig.savefig("confusion_matrix.png")`<br/>`wandb.log({"confusion_matrix": wandb.Image(fig)})`<br/>`mlflow.log_artifact("confusion_matrix.png")` |
| Compare runs | Name runs and tag model family. | `wandb.init(name="resnet50-lr1e-3", tags=["resnet", "baseline"])`<br/>`mlflow.set_tags({"model_family": "resnet", "stage": "baseline"})` |
| Resume W&B run | Continue an interrupted run. | `wandb.init(project="train", id=run_id, resume="allow")` |
| Local MLflow server | Start tracking UI locally. | `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns` |
| Promote model | Move registry version after validation. | `client.transition_model_version_stage("ChurnClassifier", version="3", stage="Production")` |
| Load best run | Query runs and load best artifact. | `runs = mlflow.search_runs(order_by=["metrics.val_loss ASC"], max_results=1)`<br/>`best_run_id = runs.iloc[0].run_id` |

## Senior experiment design

| Method | Description | Code example |
|---|---|---|
| Immutable run config | Save the exact config used by the run before training starts. | `config = OmegaConf.to_container(cfg, resolve=True)`<br/>`wandb.init(config=config)`<br/>`mlflow.log_dict(config, "config.json")` |
| Dataset fingerprint | Log dataset identity, not just dataset name. | `fingerprint = hashlib.sha256(Path("train.parquet").read_bytes()).hexdigest()`<br/>`mlflow.log_param("train_sha256", fingerprint)` |
| Code snapshot | Track commit and dirty state. | `commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()`<br/>`dirty = subprocess.call(["git", "diff", "--quiet"]) != 0`<br/>`wandb.config.update({"git_commit": commit, "git_dirty": dirty})` |
| Environment capture | Log package versions for reproducibility. | `freeze = subprocess.check_output(["pip", "freeze"], text=True)`<br/>`Path("requirements.freeze.txt").write_text(freeze)`<br/>`mlflow.log_artifact("requirements.freeze.txt")` |
| Baseline tagging | Make baselines easy to filter and compare. | `wandb.init(tags=["baseline", "xgboost", "tabular"])`<br/>`mlflow.set_tags({"kind": "baseline", "model": "xgboost"})` |
| Promotion criteria | Promote only when metrics pass explicit gates. | `if metrics["auc"] > 0.91 and metrics["latency_ms"] < 50:`<br/>`    promote_model(version)` |
| Evaluation artifact | Store predictions for audit and slice analysis. | `pred_df.to_parquet("predictions.parquet")`<br/>`mlflow.log_artifact("predictions.parquet")` |
| Run lineage | Connect preprocessing, training, and evaluation runs. | `mlflow.set_tag("parent_run_id", parent_run_id)`<br/>`wandb.config.update({"upstream_dataset_run": data_run_id})` |

## Registry and production governance

| Method | Description | Code example |
|---|---|---|
| Model signature | Record expected input/output schema with model artifact. | `signature = infer_signature(X_train, model.predict(X_train))`<br/>`mlflow.sklearn.log_model(model, "model", signature=signature)` |
| Model card artifact | Store intended use, risks, metrics, and owner. | `Path("model_card.md").write_text(model_card_text)`<br/>`mlflow.log_artifact("model_card.md")` |
| Champion challenger | Compare candidate against production model before promotion. | `delta = candidate_auc - champion_auc`<br/>`mlflow.log_metric("champion_delta_auc", delta)` |
| Stage transition audit | Add metadata when promoting a model. | `client.set_model_version_tag(name, version, "approved_by", reviewer)`<br/>`client.transition_model_version_stage(name, version, "Production")` |
| Rollback pointer | Keep previous production version discoverable. | `client.set_registered_model_alias(name, "previous-production", old_version)` |
| Artifact retention | Retain only required artifacts for cost control. | `# Configure bucket lifecycle policy for old runs and large checkpoints.` |
| PII policy | Avoid logging raw text, user IDs, or secrets into experiment trackers. | `safe_metrics = {"accuracy": acc, "loss": loss}`<br/>`wandb.log(safe_metrics)` |
| Drift monitor handoff | Log training feature stats for production drift checks. | `stats = X_train.describe().to_dict()`<br/>`mlflow.log_dict(stats, "feature_stats.json")` |

## Advanced tracking recipes

| Method | Description | Code example |
|---|---|---|
| Nested cross-validation | Keep fold metrics under one parent experiment. | `with mlflow.start_run(run_name="cv"):`<br/>`    for fold in range(5):`<br/>`        with mlflow.start_run(run_name=f"fold-{fold}", nested=True):`<br/>`            mlflow.log_metric("auc", auc)` |
| W&B table for errors | Inspect failed examples visually. | `table = wandb.Table(dataframe=errors_df)`<br/>`wandb.log({"error_analysis": table})` |
| Hyperparameter sweep guard | Stop poor runs early. | `if val_loss > baseline_loss * 1.5:`<br/>`    wandb.finish(exit_code=1)` |
| Compare by slice | Log metrics per segment. | `for segment, frame in eval_df.groupby("segment"):`<br/>`    mlflow.log_metric(f"auc_{segment}", compute_auc(frame))` |
| Load model by alias | Decouple production code from numeric versions. | `model = mlflow.pyfunc.load_model("models:/ChurnClassifier@production")` |
| Rehydrate run | Download config and artifacts for debugging. | `run = wandb.Api().run("team/project/run_id")`<br/>`config = run.config` |
