---
title: Kubernetes Basics Master Cheatsheet
sidebar_position: 17
---

# Kubernetes Basics Master Cheatsheet

## Core objects

| Method | Description | Code example |
|---|---|---|
| Pod | Smallest schedulable unit. Usually managed by a Deployment, not created directly. | `apiVersion: v1`<br/>`kind: Pod`<br/>`metadata:`<br/>`  name: ml-api`<br/>`spec:`<br/>`  containers:`<br/>`    - name: api`<br/>`      image: ml-api:1.0.0` |
| Deployment | Manages replicas, rolling updates, and self-healing pods. | `apiVersion: apps/v1`<br/>`kind: Deployment`<br/>`spec:`<br/>`  replicas: 3`<br/>`  selector:`<br/>`    matchLabels:`<br/>`      app: ml-api` |
| Service | Stable network endpoint for pods. | `apiVersion: v1`<br/>`kind: Service`<br/>`spec:`<br/>`  selector:`<br/>`    app: ml-api`<br/>`  ports:`<br/>`    - port: 80`<br/>`      targetPort: 8000` |
| Namespace | Logical cluster partition. | `kubectl create namespace ml`<br/>`kubectl get pods -n ml` |
| Job | Runs finite batch work to completion. | `apiVersion: batch/v1`<br/>`kind: Job`<br/>`spec:`<br/>`  template:`<br/>`    spec:`<br/>`      restartPolicy: Never` |
| CronJob | Runs jobs on a schedule. | `apiVersion: batch/v1`<br/>`kind: CronJob`<br/>`spec:`<br/>`  schedule: "0 2 * * *"` |

## `kubectl`

| Method | Description | Code example |
|---|---|---|
| `kubectl get` | Lists resources. | `kubectl get pods -n ml`<br/>`kubectl get deploy,svc -n ml` |
| `kubectl describe` | Shows detailed resource state and events. | `kubectl describe pod ml-api-abc123 -n ml` |
| `kubectl logs` | Reads pod logs. | `kubectl logs deploy/ml-api -n ml`<br/>`kubectl logs -f pod/ml-api-abc123 -n ml` |
| `kubectl exec` | Runs a command inside a container. | `kubectl exec -it deploy/ml-api -n ml -- bash` |
| `kubectl apply` | Creates or updates resources declaratively. | `kubectl apply -f k8s/deployment.yaml` |
| `kubectl rollout` | Inspects and controls deployment rollout. | `kubectl rollout status deploy/ml-api -n ml`<br/>`kubectl rollout undo deploy/ml-api -n ml` |

## Config, secrets, and storage

| Method | Description | Code example |
|---|---|---|
| ConfigMap | Non-secret configuration mounted as env vars or files. | `kubectl create configmap app-config --from-literal=ENV=prod -n ml` |
| Secret | Sensitive config. Use external secret managers for serious production. | `kubectl create secret generic api-secret --from-literal=TOKEN=secret -n ml` |
| Env from ConfigMap | Injects config into containers. | `envFrom:`<br/>`  - configMapRef:`<br/>`      name: app-config` |
| Env from Secret | Injects secret values into containers. | `envFrom:`<br/>`  - secretRef:`<br/>`      name: api-secret` |
| PVC | PersistentVolumeClaim requests durable storage. | `apiVersion: v1`<br/>`kind: PersistentVolumeClaim`<br/>`spec:`<br/>`  resources:`<br/>`    requests:`<br/>`      storage: 100Gi` |
| Volume mount | Mounts PVC into a container. | `volumeMounts:`<br/>`  - name: model-cache`<br/>`    mountPath: /models` |

## ML and GPU scheduling

| Method | Description | Code example |
|---|---|---|
| GPU request | Requests NVIDIA GPU resources. Requires device plugin. | `resources:`<br/>`  limits:`<br/>`    nvidia.com/gpu: 1` |
| Node selector | Schedules pods to labeled nodes. | `nodeSelector:`<br/>`  accelerator: nvidia-a100` |
| Tolerations | Allows scheduling onto tainted GPU nodes. | `tolerations:`<br/>`  - key: nvidia.com/gpu`<br/>`    operator: Exists`<br/>`    effect: NoSchedule` |
| Resource requests | Guarantees CPU and memory for stable scheduling. | `resources:`<br/>`  requests:`<br/>`    cpu: "2"`<br/>`    memory: 8Gi` |
| Readiness probe | Keeps traffic away until model is loaded. | `readinessProbe:`<br/>`  httpGet:`<br/>`    path: /ready`<br/>`    port: 8000` |
| Liveness probe | Restarts stuck containers. | `livenessProbe:`<br/>`  httpGet:`<br/>`    path: /health`<br/>`    port: 8000` |

## Helm and Kubeflow primer

| Method | Description | Code example |
|---|---|---|
| Helm install | Installs a packaged chart. | `helm repo add bitnami https://charts.bitnami.com/bitnami`<br/>`helm install redis bitnami/redis -n ml` |
| Helm values | Overrides chart configuration. | `helm upgrade --install ml-api ./chart -f values-prod.yaml` |
| Chart template | Renders manifests locally before applying. | `helm template ml-api ./chart -f values-prod.yaml` |
| Kubeflow Pipelines | Defines ML workflows as pipeline tasks. | `@dsl.pipeline(name="train-pipeline")`<br/>`def pipeline():`<br/>`    train_task = train_component()` |
| KServe | Model serving layer for Kubernetes. | `apiVersion: serving.kserve.io/v1beta1`<br/>`kind: InferenceService`<br/>`metadata:`<br/>`  name: sklearn-model` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Deploy API | Apply namespace, deployment, and service. | `kubectl apply -f namespace.yaml`<br/>`kubectl apply -f deployment.yaml`<br/>`kubectl apply -f service.yaml` |
| Port forward | Test service locally. | `kubectl port-forward svc/ml-api 8000:80 -n ml` |
| Scale deployment | Change replica count. | `kubectl scale deploy/ml-api --replicas=5 -n ml` |
| Update image | Trigger rolling update with new image tag. | `kubectl set image deploy/ml-api api=registry/ml-api:1.2.0 -n ml` |
| Debug crash loop | Inspect logs, describe pod, check events. | `kubectl logs pod/ml-api-abc -n ml --previous`<br/>`kubectl describe pod/ml-api-abc -n ml` |
| Batch training job | Run training as a Job with mounted data. | `kubectl apply -f train-job.yaml`<br/>`kubectl logs job/train-model -n ml` |
| Separate envs | Use namespaces per environment. | `kubectl create ns staging`<br/>`kubectl create ns production` |
| Delete safely | Delete resources by file or label. | `kubectl delete -f k8s/`<br/>`kubectl delete pods -l app=ml-api -n ml` |

## Senior operations and reliability

| Method | Description | Code example |
|---|---|---|
| Pod disruption budget | Maintains minimum availability during voluntary disruptions. | `apiVersion: policy/v1`<br/>`kind: PodDisruptionBudget`<br/>`spec:`<br/>`  minAvailable: 2`<br/>`  selector:`<br/>`    matchLabels:`<br/>`      app: ml-api` |
| Horizontal autoscaler | Scales replicas from CPU, memory, or custom metrics. | `kubectl autoscale deploy ml-api --cpu-percent=70 --min=2 --max=10 -n ml` |
| Rolling update strategy | Control surge and unavailable pods during deploys. | `strategy:`<br/>`  type: RollingUpdate`<br/>`  rollingUpdate:`<br/>`    maxSurge: 1`<br/>`    maxUnavailable: 0` |
| Init container | Prepare model cache or migrations before app starts. | `initContainers:`<br/>`  - name: download-model`<br/>`    image: aws-cli`<br/>`    command: ["sh", "-c", "aws s3 cp s3://bucket/model /models/model"]` |
| Security context | Run as non-root and prevent privilege escalation. | `securityContext:`<br/>`  runAsNonRoot: true`<br/>`  allowPrivilegeEscalation: false`<br/>`  readOnlyRootFilesystem: true` |
| Network policy | Restrict pod ingress/egress. | `kind: NetworkPolicy`<br/>`spec:`<br/>`  podSelector:`<br/>`    matchLabels:`<br/>`      app: ml-api`<br/>`  policyTypes: ["Ingress", "Egress"]` |
| Service account | Give workload a narrow identity. | `serviceAccountName: ml-api` |
| Resource quota | Prevent one namespace from consuming the cluster. | `kind: ResourceQuota`<br/>`spec:`<br/>`  hard:`<br/>`    requests.cpu: "100"`<br/>`    requests.memory: 400Gi` |

## ML platform patterns

| Method | Description | Code example |
|---|---|---|
| Model cache PVC | Avoid downloading large models on every restart. | `volumes:`<br/>`  - name: model-cache`<br/>`    persistentVolumeClaim:`<br/>`      claimName: model-cache-pvc` |
| Canary deployment | Route small traffic percentage to candidate model. | `# Use ingress/service mesh weights: stable 95, canary 5.` |
| Shadow traffic | Send copied requests to candidate without affecting response. | `# Mirror traffic with service mesh or gateway; log only candidate output.` |
| GPU node pool isolation | Keep GPU workloads away from CPU-only services. | `nodeSelector:`<br/>`  node-pool: gpu` |
| Priority class | Protect critical inference workloads under resource pressure. | `priorityClassName: production-inference` |
| Batch job TTL | Cleanup completed training jobs. | `spec:`<br/>`  ttlSecondsAfterFinished: 86400` |
| External secrets | Sync cloud secret manager into Kubernetes secrets. | `apiVersion: external-secrets.io/v1beta1`<br/>`kind: ExternalSecret` |
| Observability labels | Standardize labels for dashboards and cost reports. | `labels:`<br/>`  app: ml-api`<br/>`  team: ml-platform`<br/>`  model: churn` |
