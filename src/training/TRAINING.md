## Cloud Training and Hosting Options for PHI-3.5 Fine-tuning

### AWS SageMaker

#### Training Setup

```bash
# Install AWS CLI and SageMaker SDK
pip install awscli sagemaker boto3

# Configure AWS credentials
aws configure

# Create S3 bucket for data and models
aws s3 mb s3://your-phi3-training-bucket

# Upload training data to S3
aws s3 sync ./data/ s3://your-phi3-training-bucket/data/
```

#### SageMaker Training Job

```python
# Create training script for SageMaker
# File: sagemaker_train.py
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define estimator
estimator = PyTorch(
    entry_point='scripts/train_model.py',
    source_dir='.',
    role=role,
    instance_type='ml.g4dn.xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'config': 'config/training_config.yaml',
        'data-samples': 5000
    },
    environment={
        'WANDB_API_KEY': 'your-wandb-key'
    }
)

# Start training
estimator.fit({'training': 's3://your-phi3-training-bucket/data/'})
```

#### Deployment

```python
# Deploy trained model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge'
)
```

### Google Cloud Platform (Vertex AI)

#### Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

# Create GCS bucket
gsutil mb gs://your-phi3-training-bucket

# Upload data
gsutil -m cp -r ./data/ gs://your-phi3-training-bucket/
```

#### Custom Training Job

```bash
# Create Docker image for training
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "scripts/train_model.py"]
```

```bash
# Build and push container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/phi3-trainer

# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=phi3-training \
  --config=training_config.yaml \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT_ID/phi3-trainer
```

#### Vertex AI Model Deployment

```python
from google.cloud import aiplatform

# Deploy model
endpoint = aiplatform.Endpoint.create(display_name="phi3-agent-endpoint")

model = aiplatform.Model.upload(
    display_name="phi3-agent-model",
    artifact_uri="gs://your-phi3-training-bucket/models/",
    serving_container_image_uri="gcr.io/YOUR_PROJECT_ID/phi3-inference"
)

model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)
```

### Azure Machine Learning

#### Setup

```bash
# Install Azure CLI and ML extension
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az extension add -n ml

# Login and set subscription
az login
az account set --subscription YOUR_SUBSCRIPTION_ID

# Create resource group and workspace
az group create --name phi3-rg --location eastus
az ml workspace create --name phi3-workspace --resource-group phi3-rg
```

#### Training Configuration

```yaml
# azure_training.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: python scripts/train_model.py --config config/training_config.yaml
environment: azureml:pytorch-2.0-cuda11.7@latest
compute: gpu-cluster
resources:
  instance_count: 1
inputs:
  data:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/phi3-data/
outputs:
  model:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/phi3-models/
```

```bash
# Submit training job
az ml job create --file azure_training.yml --resource-group phi3-rg --workspace-name phi3-workspace
```

#### Model Deployment

```yaml
# deployment.yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: phi3-agent-endpoint
traffic:
  phi3-deployment: 100

$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: phi3-deployment
endpoint_name: phi3-agent-endpoint
model: azureml:phi3-agent-model:1
instance_type: Standard_NC6s_v3
instance_count: 1
```

```bash
# Deploy endpoint
az ml online-endpoint create --file deployment.yml
```

### Recommended Instance Types by Provider

#### AWS

- **Training**: `ml.g4dn.xlarge` (1 GPU, 16GB VRAM) - $1.20/hour
- **Training (Large)**: `ml.p3.2xlarge` (1 V100, 16GB VRAM) - $3.06/hour
- **Inference**: `ml.g4dn.xlarge` or `ml.inf1.xlarge` (cost-optimized)

#### GCP

- **Training**: `n1-standard-4` + `nvidia-tesla-t4` - $0.35/hour + $0.35/hour
- **Training (Large)**: `n1-standard-8` + `nvidia-tesla-v100` - $0.70/hour + $2.48/hour
- **Inference**: `n1-standard-2` + `nvidia-tesla-t4`

#### Azure

- **Training**: `Standard_NC4as_T4_v3` (1 T4, 16GB VRAM) - $0.526/hour
- **Training (Large)**: `Standard_NC6s_v3` (1 V100, 16GB VRAM) - $3.06/hour
- **Inference**: `Standard_NC4as_T4_v3` or `Standard_D4s_v3`

### Cost Optimization Tips

#### Spot/Preemptible Instances

```bash
# AWS Spot instances (up to 90% savings)
instance_type='ml.g4dn.xlarge'
use_spot_instances=True

# GCP Preemptible instances
machine_type='n1-standard-4'
preemptible=True

# Azure Spot instances
vm_priority='Spot'
```

#### Auto-scaling and Scheduling

```python
# Auto-shutdown after training completion
import boto3

def lambda_handler(event, context):
    sagemaker = boto3.client('sagemaker')

    # Check training job status
    response = sagemaker.describe_training_job(
        TrainingJobName=event['TrainingJobName']
    )

    if response['TrainingJobStatus'] == 'Completed':
        # Shutdown endpoint if exists
        sagemaker.delete_endpoint(EndpointName='phi3-agent-endpoint')
```

### Multi-GPU Training Setup

#### Distributed Training (4+ GPUs)

```yaml
# Update training_config.yaml for multi-GPU
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2
  dataloader_num_workers: 4
  ddp_find_unused_parameters: false

# Launch with torchrun
torchrun --nproc_per_node=4 scripts/train_model.py --config config/training_config.yaml
```

### Monitoring and Logging

#### CloudWatch (AWS)

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

# Custom metrics
cloudwatch.put_metric_data(
    Namespace='PHI3Training',
    MetricData=[
        {
            'MetricName': 'TrainingLoss',
            'Value': loss_value,
            'Unit': 'None'
        }
    ]
)
```

#### Integration with Weights & Biases

```bash
# Set environment variables for cloud training
export WANDB_API_KEY="your-wandb-key"
export WANDB_PROJECT="phi3-cloud-training"
export WANDB_ENTITY="your-team"
```

### Storage Considerations

- **Model Size**: ~7-8GB base model + ~200MB LoRA adapters
- **Training Data**: ~1-10GB depending on dataset size
- **Checkpoints**: ~200MB per checkpoint (save every 500 steps)
- **Logs**: ~100MB for full training run

Total recommended storage: **50-100GB** per training experiment
