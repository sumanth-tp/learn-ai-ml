---
title: Cloud for ML AWS and GCP Master Cheatsheet
sidebar_position: 18
---

# Cloud for ML AWS and GCP Master Cheatsheet

## Object storage

| Method | Description | Code example |
|---|---|---|
| AWS S3 upload | Stores datasets, artifacts, and model files. | `aws s3 cp model.pt s3://my-bucket/models/model.pt` |
| AWS S3 sync | Syncs local and remote directories. | `aws s3 sync ./data s3://my-bucket/data/` |
| GCS upload | Google Cloud Storage equivalent for ML artifacts. | `gcloud storage cp model.pt gs://my-bucket/models/model.pt` |
| GCS sync | Syncs directories to GCS. | `gcloud storage rsync ./data gs://my-bucket/data --recursive` |
| Python S3 client | Upload from training code. | `import boto3`<br/>`s3 = boto3.client("s3")`<br/>`s3.upload_file("model.pt", "my-bucket", "models/model.pt")` |
| Python GCS client | Upload from training code. | `from google.cloud import storage`<br/>`client = storage.Client()`<br/>`client.bucket("my-bucket").blob("models/model.pt").upload_from_filename("model.pt")` |

## Compute and GPUs

| Method | Description | Code example |
|---|---|---|
| AWS EC2 GPU | Launch GPU VMs for training or inference. | `aws ec2 run-instances --image-id ami-123 --instance-type g5.xlarge --key-name ml-key` |
| GCP GPU VM | Create GPU VM on Compute Engine. | `gcloud compute instances create trainer --machine-type=n1-standard-8 --accelerator=type=nvidia-tesla-t4,count=1` |
| SSH to VM | Connect to remote training machine. | `ssh -i key.pem ubuntu@ec2-host`<br/>`gcloud compute ssh trainer` |
| Stop VM | Avoid surprise GPU bills. | `aws ec2 stop-instances --instance-ids i-123456`<br/>`gcloud compute instances stop trainer` |
| Startup script | Bootstrap VM setup. | `#!/usr/bin/env bash`<br/>`apt-get update`<br/>`pip install -r requirements.txt` |
| GPU check | Verify driver visibility. | `nvidia-smi` |

## Managed ML platforms

| Method | Description | Code example |
|---|---|---|
| SageMaker training | Managed AWS training jobs with images, data channels, and output paths. | `estimator = PyTorch(entry_point="train.py", role=role, instance_type="ml.g5.xlarge", instance_count=1)`<br/>`estimator.fit({"train": "s3://bucket/train/"})` |
| SageMaker endpoint | Deploy trained model for online inference. | `predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")` |
| Vertex AI training | Managed GCP custom training jobs. | `job = aiplatform.CustomTrainingJob(display_name="train", script_path="train.py", container_uri=image_uri)`<br/>`job.run(machine_type="n1-standard-8")` |
| Vertex AI endpoint | Deploy model to endpoint. | `model = aiplatform.Model.upload(display_name="model", artifact_uri="gs://bucket/model", serving_container_image_uri=image_uri)`<br/>`endpoint = model.deploy(machine_type="n1-standard-4")` |
| Batch prediction | Run offline inference over large datasets. | `model.batch_predict(job_display_name="batch", gcs_source="gs://bucket/input", gcs_destination_prefix="gs://bucket/output")` |

## IAM, secrets, and networking

| Method | Description | Code example |
|---|---|---|
| AWS IAM role | Prefer roles over long-lived access keys on EC2/SageMaker. | `aws iam create-role --role-name ml-training-role --assume-role-policy-document file://trust.json` |
| GCP service account | Identity for workloads. | `gcloud iam service-accounts create ml-runner` |
| Least privilege | Grant only required storage/model permissions. | `gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:ml-runner@PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer` |
| AWS Secrets Manager | Store API keys and credentials. | `aws secretsmanager get-secret-value --secret-id openai-api-key` |
| GCP Secret Manager | Store and access secrets. | `gcloud secrets versions access latest --secret=openai-api-key` |
| Private networking | Keep databases and endpoints off the public internet where possible. | `aws ec2 describe-security-groups`<br/>`gcloud compute networks list` |

## Deploying models

| Method | Description | Code example |
|---|---|---|
| Container registry AWS | Push model API image to ECR. | `aws ecr get-login-password --region us-east-1`<br/>`docker tag ml-api ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ml-api:1.0.0`<br/>`docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ml-api:1.0.0` |
| Container registry GCP | Push image to Artifact Registry. | `gcloud auth configure-docker us-central1-docker.pkg.dev`<br/>`docker push us-central1-docker.pkg.dev/PROJECT/ml/ml-api:1.0.0` |
| Cloud Run | Serverless container deployment on GCP. | `gcloud run deploy ml-api --image us-central1-docker.pkg.dev/PROJECT/ml/ml-api:1.0.0 --region us-central1` |
| ECS service | Run containers on AWS ECS. | `aws ecs update-service --cluster ml --service ml-api --force-new-deployment` |
| Scheduled batch | Run batch jobs on a schedule. | `aws events put-rule --schedule-expression "rate(1 day)"`<br/>`gcloud scheduler jobs create http daily-batch --schedule="0 2 * * *"` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Train on cloud, log artifacts | Save checkpoints to object storage. | `checkpoint = "model.pt"`<br/>`torch.save(model.state_dict(), checkpoint)`<br/>`s3.upload_file(checkpoint, bucket, "runs/123/model.pt")` |
| Use preemptible or spot | Reduce cost for fault-tolerant jobs. | `gcloud compute instances create trainer --preemptible` |
| Externalize config | Use env vars for bucket, model path, region. | `MODEL_URI=s3://bucket/models/model.pt python serve.py` |
| Tag resources | Track ownership and cost center. | `aws ec2 create-tags --resources i-123 --tags Key=Project,Value=ml` |
| Lifecycle policies | Expire old artifacts automatically. | `aws s3api put-bucket-lifecycle-configuration --bucket my-bucket --lifecycle-configuration file://lifecycle.json` |
| Model promotion | Copy validated model to production path. | `aws s3 cp s3://bucket/staging/model.pt s3://bucket/prod/model.pt` |
| Cost guardrail | Stop idle instances. | `gcloud compute instances stop trainer --zone us-central1-a` |
| Reproducible job | Store image tag, data URI, code commit, and config. | `metadata = {"image": image, "data": data_uri, "commit": commit, "config": config}` |

## Senior cloud architecture

| Method | Description | Code example |
|---|---|---|
| Landing zone separation | Separate dev, staging, and prod accounts/projects for blast-radius control. | `aws organizations list-accounts`<br/>`gcloud projects list` |
| Workload identity | Prefer short-lived workload identity over static keys. | `# AWS: IAM role for service account`<br/>`# GCP: Workload Identity Federation` |
| Private artifact path | Keep model artifacts in private buckets with least-privilege roles. | `aws s3api put-bucket-public-access-block --bucket ml-prod --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true` |
| VPC endpoint | Access object storage privately where supported. | `aws ec2 create-vpc-endpoint --vpc-id vpc-123 --service-name com.amazonaws.us-east-1.s3` |
| KMS encryption | Encrypt artifacts with managed keys and audit access. | `aws s3 cp model.pt s3://bucket/model.pt --sse aws:kms --sse-kms-key-id alias/ml` |
| Budget alerts | Alert before cost surprises. | `aws budgets create-budget --account-id 123 --budget file://budget.json --notifications-with-subscribers file://notify.json` |
| IaC modules | Use Terraform/Pulumi modules for reproducible infra. | `terraform plan -var-file=prod.tfvars`<br/>`terraform apply -var-file=prod.tfvars` |
| Cross-region DR | Replicate critical artifacts and define recovery objectives. | `aws s3api put-bucket-replication --bucket ml-prod --replication-configuration file://replication.json` |

## Production ML deployment decisions

| Method | Description | Code example |
|---|---|---|
| Online endpoint | Low-latency synchronous inference. | `# Use SageMaker endpoint, Vertex endpoint, ECS, Cloud Run, or Kubernetes depending on latency and control needs.` |
| Batch inference | Large offline scoring with cheaper async compute. | `python batch_score.py --input s3://bucket/input --output s3://bucket/output` |
| Feature store boundary | Avoid training-serving skew by sharing feature definitions. | `features = feature_store.get_online_features(entity_rows=entities, features=feature_refs)` |
| Blue-green deploy | Deploy new stack beside old stack and switch traffic. | `# Create green endpoint, validate, then update DNS/load balancer target.` |
| Model rollback | Keep previous model URI and image tag. | `MODEL_URI=s3://bucket/models/churn/v6/model.pt`<br/>`IMAGE_TAG=ml-api:1.6.0` |
| Audit log | Store who deployed what, when, and why. | `deploy_record = {"user": user, "model": model_uri, "image": image, "commit": commit}` |
| Data residency | Keep data and model artifacts in approved regions. | `AWS_REGION=eu-west-1`<br/>`GOOGLE_CLOUD_LOCATION=europe-west4` |
| Quota planning | Request GPU/endpoint quota before launch. | `gcloud compute regions describe us-central1`<br/>`aws service-quotas list-service-quotas --service-code ec2` |
