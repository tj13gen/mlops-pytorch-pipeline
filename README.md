# MLOps PyTorch Pipeline

A production-ready PyTorch image classification pipeline featuring containerized model training with Docker, orchestrating batch jobs and real-time model inference on Kubernetes, complete with Git branching workflows and CI verification.

---

## 🏗 Architecture Overview

![Architecture Diagram](architecture_diagram.png)

### System Components

| Component | Type | Responsibility |
| :--- | :--- | :--- |
| **ConfigMap** | `v1/ConfigMap` | Decoupled training hyperparameters (batch size, learning rate, epochs) |
| **Persistent Storage** | `PersistentVolumeClaim` | Shared storage for CIFAR-10 data cache and exported model weights |
| **Training Job** | `batch/v1 Job` | ResNet-18 batch training with early stopping and GPU acceleration |
| **Model Serving** | `apps/v1 Deployment` | 2-replica FastAPI inference service with rolling update & health probes |
| **ClusterIP Service** | `v1/Service` | Internal load-balanced network gateway on port 80 -> 8080 |
| **Auto-Scaler (HPA)** | `autoscaling/v2` | Dynamic replica scaling (2 to 10 pods) based on 80% CPU utilization |

---

## 📁 Repository Structure

```
mlops-pytorch-pipeline/
|-- README.md                         # Project documentation & architecture overview
|-- MLOps_Assignment_Report.pdf       # Consolidated submission report
|-- .gitignore                        # Git exclusion rules
|-- .github/
|   \-- workflows/
|       \-- ci.yml                    # Automated CI testing and Docker build pipeline
|-- src/
|   |-- train.py                      # Training loop with JSON logging & early stopping
|   |-- model.py                      # ResNet-18 model architecture
|   |-- dataset.py                    # Data loader with augmentation pipelines
|   \-- serve.py                      # FastAPI inference API with health checks
|-- configs/
|   \-- training_config.yaml          # YAML training hyperparameters
|-- docker/
|   |-- Dockerfile.train              # Multi-stage training image
|   \-- Dockerfile.serve              # Hardened, non-root serving runtime
|-- k8s/
|   |-- namespace.yaml                # Dedicated ml-training namespace
|   |-- configmap.yaml                # Decoupled training settings
|   |-- training-job.yaml             # Kubernetes Job with PVC & GPU tolerations
|   |-- serving-deployment.yaml       # High-availability Serving Deployment
|   |-- serving-service.yaml          # ClusterIP service exposing prediction endpoint
|   \-- hpa.yaml                      # Horizontal Pod Autoscaler
|-- requirements/
|   |-- train.txt                     # Pinned training dependencies
|   \-- serve.txt                     # Minimal inference dependencies
\-- tests/
    \-- test_model.py                 # PyTest test suite
```

---

## 🚀 Setup & Local Execution Guide

### Prerequisites
- Python 3.10+
- Docker Desktop
- `kubectl` CLI & Minikube/kind Kubernetes cluster

### 1. Local Training & Serving Verification
```bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run containerized training with mounted data/checkpoint directories
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving container
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test inference endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

### 2. Kubernetes Deployment
```bash
# Apply namespace, configmap, and training job
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# Monitor training job execution
kubectl get jobs -n ml-training -w

# Deploy serving deployment, service, and HPA
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Port-forward service to port 8080 for testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

---

## 📝 Reflection Write-up

### What was the most challenging part?

The primary technical challenge in implementing this end-to-end PyTorch MLOps pipeline lay in harmonizing state persistence, container lifecycles, and configuration consistency across training batch workloads and real-time inference services in Kubernetes.

In a containerized environment, training workloads operate as transient batch Jobs requiring write access to model weights and dataset caches, while serving deployments demand high-availability, read-only access to checkpoint artifacts with continuous health monitoring. Designing the shared volume architecture with appropriate `PersistentVolumeClaim` access modes and directory mounts (`/app/checkpoints` and `/app/data`) required strict coordination to ensure that the serving replicas never attempted to read incomplete checkpoint files during active training epochs.

Another significant hurdle involved configuring resilient health check orchestration and zero-downtime rolling updates (`maxSurge: 1`, `maxUnavailable: 0`). Because deep learning models take non-trivial time to load weights into memory and initialize compute graphs, standard HTTP health probes can easily trigger false failure loops if not tuned properly. By configuring an initial delay of 15 seconds and setting a 5-second polling interval on the `/health` Readiness probe, we ensured that the Kubernetes Service traffic router only directs user inference requests to pods that have fully hydrated the PyTorch model checkpoint into memory.

Additionally, optimizing multi-stage Docker builds presented subtle challenges in image footprint reduction and security hardening. Separating heavy training build toolchains from lightweight inference runtimes reduced attack surfaces and image pull times. Transitioning the serving container to run under an unprivileged `appuser` (UID 1000) while still retaining permissions to read mounted checkpoint volumes required explicit permission handling in the Docker build stages and Kubernetes pod security contexts.

Finally, managing the configuration lifecycle via Kubernetes `ConfigMap` resources ensured that hyperparameters like learning rate, batch size, and early stopping patience could be modified dynamically without rebuilding container images. Structuring clean Git feature-branch workflows with automated CI testing ensured that model architectures, Dockerfiles, and Kubernetes manifests remained reproducible, robust, and production-ready.
