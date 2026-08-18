# MLOps PyTorch Pipeline

A production-ready PyTorch image classification pipeline featuring containerized model training with Docker, orchestrating batch jobs and real-time model inference on Kubernetes, complete with Git branching workflows and CI verification.

---

## 🏗 Architecture Diagram

```
                       +-------------------------------------------------------+
                       |              Kubernetes Cluster (ml-training)         |
                       |                                                       |
                       |  +---------------------+      +--------------------+  |
                       |  | ConfigMap           |      | PersistentVolume   |  |
                       |  | training-config     |      | /app/checkpoints   |  |
                       |  +----------+----------+      +---------+----------+  |
                       |             |                           |             |
                       |             v                           v             |
                       |  +----------+----------+      +---------+----------+  |
                       |  | Training Job        |      | Serving Deployment |  |
                       |  | (ResNet-18 Job)     |----> | (FastAPI Replicas) |  |
                       |  +---------------------+      +----------+---------+  |
                       |                                          |            |
                       |                                          v            |
                       |                               +----------+---------+  |
                       |                               | ClusterIP Service  |  |
                       |                               | (Port 80 -> 8080)  |  |
                       |                               +----------+---------+  |
                       +------------------------------------------|------------+
                                                                  |
                                                           [ curl /predict ]
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

The primary technical challenge in implementing this end-to-end PyTorch MLOps pipeline lay in harmonizing state persistence across container boundaries between separate training batch workloads and real-time model inference services. In Kubernetes environments, training workloads operate as transient batch Jobs requiring full write access to model weights and dataset caches, while serving deployments demand high-availability, read-only access to checkpoint artifacts with continuous health monitoring. 

Designing multi-stage Docker builds required careful optimization to isolate training dependencies (such as heavy build toolchains) from lightweight inference runtimes. Ensuring zero-downtime rolling updates (`maxSurge: 1`, `maxUnavailable: 0`) alongside stringent Liveness (`/health` probe every 10s) and Readiness probes required strict synchronization: the serving containers needed to gracefully delay readiness until model checkpoints were fully hydrated into memory from persistent storage. Furthermore, structuring clean Git feature-branch workflows with automated CI checks ensured that both model artifacts and Kubernetes manifests remained reproducible, robust, and production-ready.
