# SLM Agent Deployment Guide

Complete guide for building and deploying the SLM Agent with its web UI.

## Quick Start

### Local Development (Without Docker)

1. **Start the Backend Server:**

```bash
# From the inference directory
python server.py --model-info model/microsoft/Phi-3.5-mini-instruct/model.json --host 0.0.0.0 --port 8000
```

2. **Start the UI Development Server:**

```bash
# From the ui directory
cd ui
npm install
npm run dev
```

The UI will be at `http://localhost:3000`, proxying API requests to `http://localhost:8000`.

### Docker Deployment (Production)

**Build the complete image:**

```bash
# From the inference directory
docker build -t slm-agent:latest .
```

**Run the container:**

```bash
docker run -p 8000:8000 --gpus all \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/cache:/app/cache \
  slm-agent:latest
```

Access both UI and API at `http://localhost:8000`.

## Build Process

### Multi-Stage Docker Build

The Dockerfile uses a multi-stage build:

1. **Stage 1 (ui-builder):** Builds the React/TypeScript UI

   - Uses Node.js 18 Alpine image
   - Installs npm dependencies
   - Runs `npm run build` to create optimized production bundle
   - Output: `/ui/dist` directory

2. **Stage 2 (main):** Creates the Python/CUDA application
   - Uses NVIDIA CUDA base image
   - Installs Python and dependencies
   - Copies built UI from Stage 1 to `/app/ui/dist`
   - Configures Flask to serve UI static files

### Manual Build Steps

If you need to build without Docker:

**1. Build the UI:**

```bash
cd ui
npm install
npm run build
# Output will be in ui/dist/
```

**2. Run the server with UI:**

```bash
cd ..
python server.py \
  --model-info model/microsoft/Phi-3.5-mini-instruct/model.json \
  --ui-dist-path ./ui/dist \
  --host 0.0.0.0 \
  --port 8000
```

## Configuration

### Environment Variables

```bash
# Model configuration
MODEL_PATH=/app/model
MODEL_NAME=Phi-3.5-mini-instruct

# Server configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# CUDA configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Cache directories
HF_HOME=/app/cache/huggingface
TRANSFORMERS_CACHE=/app/cache/huggingface/transformers

# UI configuration
UI_DIST_PATH=/app/ui/dist
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  slm-agent:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model:ro
      - ./cache:/app/cache
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/model
      - DEVICE=cuda
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## Deployment Scenarios

### Development

**UI Hot Reload:**

```bash
# Terminal 1: Backend
python server.py --model-info model/microsoft/Phi-3.5-mini-instruct/model.json

# Terminal 2: UI with hot reload
cd ui && npm run dev
```

### Production on Single Server

**Option 1: Docker (Recommended)**

```bash
docker build -t slm-agent:latest .
docker run -d -p 8000:8000 --gpus all \
  --name slm-agent \
  -v /path/to/models:/app/model:ro \
  -v /path/to/cache:/app/cache \
  --restart unless-stopped \
  slm-agent:latest
```

**Option 2: Systemd Service**

Create `/etc/systemd/system/slm-agent.service`:

```ini
[Unit]
Description=SLM Agent Service
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/slm-agent/inference
Environment="PATH=/opt/slm-agent/venv/bin:$PATH"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/slm-agent/venv/bin/python server.py \
  --model-info model/microsoft/Phi-3.5-mini-instruct/model.json \
  --ui-dist-path ui/dist \
  --host 0.0.0.0 \
  --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable slm-agent
sudo systemctl start slm-agent
```

### Production with Reverse Proxy

**Nginx Configuration:**

```nginx
upstream slm_agent {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy to SLM Agent
    location / {
        proxy_pass http://slm_agent;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts for long-running model inference
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }

    # Cache static UI assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        proxy_pass http://slm_agent;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: slm-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: slm-agent
  template:
    metadata:
      labels:
        app: slm-agent
    spec:
      containers:
        - name: slm-agent
          image: slm-agent:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "16Gi"
              cpu: "4"
            requests:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "2"
          volumeMounts:
            - name: model-storage
              mountPath: /app/model
              readOnly: true
            - name: cache-storage
              mountPath: /app/cache
          env:
            - name: DEVICE
              value: "cuda"
            - name: LOG_LEVEL
              value: "INFO"
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: model-pvc
        - name: cache-storage
          persistentVolumeClaim:
            claimName: cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: slm-agent-service
spec:
  selector:
    app: slm-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

Apply:

```bash
kubectl apply -f k8s-deployment.yaml
```

## Monitoring and Logging

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "model_loaded": true
}
```

### Docker Logs

```bash
# Follow logs
docker logs -f slm-agent

# Last 100 lines
docker logs --tail 100 slm-agent
```

### Resource Monitoring

```bash
# GPU usage
nvidia-smi -l 1

# Docker stats
docker stats slm-agent

# System resources
htop
```

## Troubleshooting

### UI Not Loading

1. **Check if UI was built:**

```bash
ls -la ui/dist/
# Should contain index.html and assets/
```

2. **Verify UI path in container:**

```bash
docker exec slm-agent ls -la /app/ui/dist/
```

3. **Check server logs:**

```bash
docker logs slm-agent | grep "Serving UI"
```

### API Connection Issues

1. **Verify server is running:**

```bash
curl http://localhost:8000/health
```

2. **Check CORS settings:**

```bash
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS \
  http://localhost:8000/chat
```

### GPU Not Detected

1. **Verify NVIDIA drivers:**

```bash
nvidia-smi
```

2. **Check Docker GPU access:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

3. **Install NVIDIA Container Toolkit:**

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Model Loading Errors

1. **Check model files exist:**

```bash
ls -la model/microsoft/Phi-3.5-mini-instruct/
```

2. **Verify model.json configuration:**

```bash
cat model/microsoft/Phi-3.5-mini-instruct/model.json
```

3. **Check memory requirements:**

```bash
# Model requires ~8GB GPU memory
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

## Performance Optimization

### UI Build Optimization

```bash
# Production build with size analysis
cd ui
npm run build -- --mode production

# Check bundle size
du -sh dist/
```

### Backend Optimization

1. **Use flash-attention for faster inference**
2. **Enable model quantization (INT8/INT4)**
3. **Adjust batch sizes based on GPU memory**
4. **Use gunicorn for production Flask serving:**

```bash
pip install gunicorn

gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 server:app
```

## Security Considerations

1. **Use HTTPS in production** (via nginx/reverse proxy)
2. **Implement authentication** (JWT tokens, OAuth)
3. **Rate limiting** for API endpoints
4. **Input validation** for user messages
5. **Network isolation** for model server
6. **Regular security updates** for dependencies

## Backup and Recovery

```bash
# Backup model cache
tar -czf cache-backup.tar.gz cache/

# Backup session data (if persisted)
docker exec slm-agent tar -czf - /app/data > data-backup.tar.gz

# Restore
docker exec -i slm-agent tar -xzf - -C /app/data < data-backup.tar.gz
```

## Scaling

For high-traffic deployments:

1. **Multiple replicas** behind load balancer
2. **Separate API and UI** services
3. **Redis** for session storage
4. **Message queue** (RabbitMQ/Celery) for async processing
5. **Model serving framework** (TorchServe, Triton)

## Support

For issues and questions:

- Check logs: `docker logs slm-agent`
- Review configuration: `docker inspect slm-agent`
- Test endpoints: `curl http://localhost:8000/info`
