# Part B: Data Engineering Pipeline - Deployment Guide

## RTV Senior Data Scientist Technical Assessment

---

## 🚀 Quick Start Deployment

### Prerequisites

- Python 3.8+
- Docker (optional)
- AWS/Azure account (for production)
- PostgreSQL 12+ (for production)
- Redis (for background processing)
- Minimum 8GB RAM (for ML models)
- 4+ CPU cores (recommended)

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd part_b_data_engineering

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy configuration template
cp config/pipeline_config.py.example config/pipeline_config.py

# Edit configuration for your environment
# Set storage provider, database credentials, ML model paths, etc.
```

### 3. Database Setup

```sql
-- For PostgreSQL (Metadata)
CREATE DATABASE rtv_pipeline;
CREATE USER pipeline_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rtv_pipeline TO pipeline_user;

-- For SQLite (Monitoring) - automatically created
-- File: monitoring/pipeline_metrics.db
```

### 4. Model Setup

```bash
# Ensure Part A models are available
ls ../part_a_predictive_modeling/models/
# Should contain: vulnerability_model.pkl, feature_pipeline.pkl

# Or use demo models
python -c "
from pipeline.model_trainer import ModelTrainer
trainer = ModelTrainer()
print('Model status:', trainer.get_model_info()['status'])
"
```

### 5. Run Complete Integration Demo

```bash
# Run comprehensive ML-integrated demonstration
python pipeline_demo.py

# Expected output:
# - Feature engineering: 50+ features processed
# - Model training: 97.9% accuracy maintained
# - ETL orchestration: Complete pipeline demo
# - Data validation: Quality checks passed
# - Real-time predictions: Vulnerability assessments generated
```

### 6. Start Monitoring

```bash
# Initialize monitoring database
python -c "
from monitoring.pipeline_monitor import PipelineMonitor
import asyncio
asyncio.run(PipelineMonitor().initialize())
"

# Check system health
python -c "
from monitoring.pipeline_monitor import PipelineMonitor
monitor = PipelineMonitor()
health = monitor.get_pipeline_health()
print(f'Health score: {health[\"health_score\"]:.2f}')
print(f'Status: {health[\"status\"]}')
"
```

---

## 🏗️ Production Deployment

### Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rtv_pipeline
      - STORAGE_PROVIDER=aws
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - MODEL_PATH=models/production_model.pkl
      - MONITORING_DB_PATH=monitoring/pipeline_metrics.db
      - MIN_NEW_SAMPLES=1000
      - PERFORMANCE_THRESHOLD=0.85
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./monitoring:/app/monitoring
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=rtv_pipeline
      - POSTGRES_USER=pipeline_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  celery-worker:
    build: .
    command: celery -A pipeline.etl_orchestrator worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rtv_pipeline
      - MODEL_PATH=models/production_model.pkl
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./monitoring:/app/monitoring
    depends_on:
      - postgres
      - redis

  scheduler:
    build: .
    command: python -c "from pipeline.etl_orchestrator import create_etl_orchestrator; import asyncio; asyncio.run(create_etl_orchestrator().start_pipeline())"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rtv_pipeline
      - MODEL_PATH=models/production_model.pkl
      - MONITORING_DB_PATH=monitoring/pipeline_metrics.db
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./monitoring:/app/monitoring
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis

  monitoring:
    build: .
    command: python -c "from monitoring.pipeline_monitor import PipelineMonitor; import asyncio; monitor = PipelineMonitor(); asyncio.run(monitor.start_monitoring())"
    environment:
      - MONITORING_DB_PATH=monitoring/pipeline_metrics.db
      - ALERT_EMAIL=admin@yourorg.com
    volumes:
      - ./monitoring:/app/monitoring
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtv-pipeline-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rtv-pipeline-api
  template:
    metadata:
      labels:
        app: rtv-pipeline-api
    spec:
      containers:
        - name: api
          image: rtv/pipeline:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: database-url
            - name: MODEL_PATH
              value: "models/production_model.pkl"
            - name: MONITORING_DB_PATH
              value: "monitoring/pipeline_metrics.db"
            - name: PERFORMANCE_THRESHOLD
              value: "0.85"
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
            - name: models-volume
              mountPath: /app/models
            - name: monitoring-volume
              mountPath: /app/monitoring
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: rtv-data-pvc
        - name: models-volume
          persistentVolumeClaim:
            claimName: rtv-models-pvc
        - name: monitoring-volume
          persistentVolumeClaim:
            claimName: rtv-monitoring-pvc

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtv-pipeline-scheduler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rtv-pipeline-scheduler
  template:
    metadata:
      labels:
        app: rtv-pipeline-scheduler
    spec:
      containers:
        - name: scheduler
          image: rtv/pipeline:latest
          command:
            [
              "python",
              "-c",
              "from pipeline.etl_orchestrator import create_etl_orchestrator; import asyncio; asyncio.run(create_etl_orchestrator().start_pipeline())",
            ]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: database-url
            - name: MODEL_PATH
              value: "models/production_model.pkl"
            - name: MONITORING_DB_PATH
              value: "monitoring/pipeline_metrics.db"
          volumeMounts:
            - name: data-volume
              mountPath: /app/data
            - name: models-volume
              mountPath: /app/models
            - name: monitoring-volume
              mountPath: /app/monitoring
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
      volumes:
        - name: data-volume
          persistentVolumeClaim:
            claimName: rtv-data-pvc
        - name: models-volume
          persistentVolumeClaim:
            claimName: rtv-models-pvc
        - name: monitoring-volume
          persistentVolumeClaim:
            claimName: rtv-monitoring-pvc

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtv-pipeline-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rtv-pipeline-monitor
  template:
    metadata:
      labels:
        app: rtv-pipeline-monitor
    spec:
      containers:
        - name: monitor
          image: rtv/pipeline:latest
          command:
            [
              "python",
              "-c",
              "from monitoring.pipeline_monitor import PipelineMonitor; import asyncio; monitor = PipelineMonitor(); asyncio.run(monitor.start_monitoring())",
            ]
          env:
            - name: MONITORING_DB_PATH
              value: "monitoring/pipeline_metrics.db"
            - name: ALERT_EMAIL
              valueFrom:
                secretKeyRef:
                  name: pipeline-secrets
                  key: alert-email
          volumeMounts:
            - name: monitoring-volume
              mountPath: /app/monitoring
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "200m"
      volumes:
        - name: monitoring-volume
          persistentVolumeClaim:
            claimName: rtv-monitoring-pvc
```

---

## 🔧 Configuration Guide

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/rtv_pipeline
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Storage Configuration
STORAGE_PROVIDER=aws  # aws, azure, or local
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_BUCKET_NAME=rtv-pipeline-data
AWS_REGION=us-east-1

# ML Configuration
MODEL_PATH=models/production_model.pkl
MIN_NEW_SAMPLES=1000
PERFORMANCE_THRESHOLD=0.85
FEATURE_ENGINEERING_CACHE=true
RETRAIN_SCHEDULE="0 2 * * 0"  # Weekly on Sunday at 2 AM

# Monitoring Configuration
MONITORING_DB_PATH=monitoring/pipeline_metrics.db
HEALTH_CHECK_INTERVAL=300  # 5 minutes
ALERT_EMAIL=admin@yourorg.com
ALERT_THRESHOLDS='{"pipeline_failure_rate": 0.05, "data_quality_score": 0.95, "model_accuracy": 0.90}'

# Security Configuration
SECRET_KEY=your-256-bit-secret-key
ENCRYPTION_KEY=your-fernet-encryption-key
JWT_EXPIRATION_HOURS=24

# Processing Configuration
MAX_WORKERS=4
BATCH_SIZE=1000
VALIDATION_STRICT_MODE=true
ASYNC_PROCESSING=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/pipeline.log
```

### ML Model Configuration

```python
# config/pipeline_config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MLConfig:
    model_path: str = "models/production_model.pkl"
    feature_pipeline_path: str = "models/feature_pipeline.pkl"
    min_new_samples: int = 1000
    performance_threshold: float = 0.85
    retrain_schedule: str = "0 2 * * 0"  # Weekly
    model_metrics_threshold: Dict[str, float] = None
    auto_deploy: bool = True

    def __post_init__(self):
        if self.model_metrics_threshold is None:
            self.model_metrics_threshold = {
                'accuracy': 0.90,
                'f1_score': 0.90,
                'precision': 0.85,
                'recall': 0.85
            }

@dataclass
class MonitoringConfig:
    db_path: str = "monitoring/pipeline_metrics.db"
    health_check_interval: int = 300  # 5 minutes
    alert_email: str = None
    alert_thresholds: Dict[str, float] = None
    system_metrics_enabled: bool = True

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'pipeline_failure_rate': 0.05,
                'data_quality_score': 0.95,
                'model_accuracy': 0.90,
                'cpu_usage': 0.85,
                'memory_usage': 0.90,
                'disk_usage': 0.85
            }

@dataclass
class FeatureEngineeringConfig:
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    parallel_processing: bool = True
    validation_enabled: bool = True
    preprocessing_pipeline_path: str = "models/preprocessing_pipeline.pkl"
```

### Cloud Provider Setup

#### AWS Configuration

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Create S3 bucket for pipeline data
aws s3 mb s3://rtv-pipeline-data

# Set up IAM permissions for pipeline access
aws iam create-policy --policy-name RTVPipelinePolicy --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::rtv-pipeline-data",
        "arn:aws:s3:::rtv-pipeline-data/*"
      ]
    }
  ]
}'
```

#### Azure Configuration

```bash
# Install Azure CLI
pip install azure-cli

# Login to Azure
az login

# Create resource group
az group create --name rtv-pipeline --location eastus

# Create storage account
az storage account create \
  --name rtvpipelinestorage \
  --resource-group rtv-pipeline \
  --location eastus \
  --sku Standard_LRS

# Create container for pipeline data
az storage container create \
  --name pipeline-data \
  --account-name rtvpipelinestorage
```

---

## 📊 Monitoring Setup

### Health Checks

```bash
# System health endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "health_score": 0.96,
  "timestamp": "2024-12-15T10:30:00Z",
  "components": {
    "database": "healthy",
    "storage": "healthy",
    "model_service": "healthy",
    "feature_engineering": "healthy",
    "monitoring": "healthy"
  }
}

# Readiness check
curl http://localhost:8000/ready

# Pipeline status
curl http://localhost:8000/pipeline/status
```

### Monitoring Database

```bash
# Check monitoring database
sqlite3 monitoring/pipeline_metrics.db

# View recent pipeline runs
.tables
SELECT * FROM pipeline_runs ORDER BY start_time DESC LIMIT 5;

# View data quality metrics
SELECT * FROM data_quality_metrics ORDER BY timestamp DESC LIMIT 10;

# View model performance
SELECT * FROM model_performance_metrics ORDER BY timestamp DESC LIMIT 10;

# View system metrics
SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 10;

# View active alerts
SELECT * FROM alerts WHERE resolved = FALSE;
```

### Alerting Configuration

```python
# monitoring/alert_config.py
ALERT_RULES = {
    'pipeline_failure': {
        'condition': 'failure_rate > 0.05',
        'severity': 'ERROR',
        'notification_channels': ['email', 'slack']
    },
    'data_quality_degradation': {
        'condition': 'quality_score < 0.95',
        'severity': 'WARNING',
        'notification_channels': ['email']
    },
    'model_performance_degradation': {
        'condition': 'accuracy < 0.90',
        'severity': 'ERROR',
        'notification_channels': ['email', 'slack']
    },
    'high_resource_usage': {
        'condition': 'cpu_usage > 0.85 OR memory_usage > 0.90',
        'severity': 'WARNING',
        'notification_channels': ['email']
    }
}

EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'alerts@yourorg.com',
    'password': 'your_password',
    'recipients': ['admin@yourorg.com', 'data-team@yourorg.com']
}
```

---

## 🧪 Testing Deployment

### Unit Tests

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific component tests
pytest tests/test_feature_engineer.py -v
pytest tests/test_model_trainer.py -v
pytest tests/test_pipeline_monitor.py -v

# Expected coverage targets:
# Feature Engineering: 95%+
# Model Training: 95%+
# Pipeline Monitoring: 90%+
# Data Validation: 95%+
# Storage Operations: 90%+
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v --slow

# Test scenarios include:
# - End-to-end pipeline execution
# - ML model training and inference
# - Monitoring and alerting
# - Storage operations across providers
# - Error handling and recovery
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000

# Performance targets:
# - 1000 concurrent users
# - 800+ records/minute throughput
# - < 5 second response time
# - < 0.1% error rate
# - ML inference: < 100ms per record
```

### Smoke Tests

```bash
# Quick deployment validation
python tests/smoke_test.py

# This will:
# 1. Check all service endpoints
# 2. Validate database connections
# 3. Test storage connectivity
# 4. Verify ML model loading
# 5. Check monitoring system
# 6. Validate feature engineering pipeline
```

---

## 🔧 Troubleshooting

### Common Deployment Issues

#### 1. Model Loading Failures

```bash
# Check model files exist
ls -la models/
# Should contain: vulnerability_model.pkl, feature_pipeline.pkl

# Test model loading
python -c "
from pipeline.model_trainer import ModelTrainer
trainer = ModelTrainer()
info = trainer.get_model_info()
print('Model status:', info['status'])
print('Model accuracy:', info.get('accuracy', 'N/A'))
"

# If models missing, copy from Part A
cp ../part_a_predictive_modeling/models/* models/
```

#### 2. Monitoring Database Issues

```bash
# Check monitoring database exists and is accessible
ls -la monitoring/
sqlite3 monitoring/pipeline_metrics.db ".tables"

# Reinitialize if corrupted
python -c "
from monitoring.pipeline_monitor import PipelineMonitor
import asyncio
monitor = PipelineMonitor()
asyncio.run(monitor.initialize())
print('Monitoring database reinitialized')
"
```

#### 3. Feature Engineering Failures

```bash
# Test feature engineering
python -c "
from pipeline.feature_engineer import FeatureEngineer
import pandas as pd

# Create sample data
sample_data = pd.DataFrame({
    'household_size': [5],
    'total_income': [25000],
    'has_electricity': [1],
    'district': ['Kitgum']
})

engineer = FeatureEngineer()
try:
    result = engineer.transform_data(sample_data)
    print('Feature engineering: OK')
    print(f'Features generated: {len(result.columns)}')
except Exception as e:
    print(f'Feature engineering failed: {e}')
"
```

#### 4. High Memory Usage

```bash
# Monitor memory usage
htop  # or top

# Optimize for production
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Reduce batch sizes if needed
# In config/pipeline_config.py:
# config.processing.batch_size = 500  # Reduce from 1000
```

#### 5. Storage Connection Issues

```bash
# Test storage connectivity
python -c "
from storage.storage_manager import StorageManager
import asyncio
import pandas as pd

async def test_storage():
    storage = StorageManager()
    try:
        # Test basic operations
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        await storage.store_data(test_data, 'test/connectivity.parquet')
        retrieved = await storage.retrieve_data('test/connectivity.parquet')
        print('Storage connectivity: OK')
    except Exception as e:
        print(f'Storage connectivity failed: {e}')

asyncio.run(test_storage())
"
```

---

## 📈 Performance Optimization

### Production Tuning

```python
# config/performance_config.py
PERFORMANCE_CONFIG = {
    # Database optimizations
    'db_pool_size': 20,
    'db_pool_timeout': 30,
    'db_pool_recycle': 3600,

    # Feature engineering optimizations
    'feature_batch_size': 1000,
    'feature_parallel_workers': 4,
    'feature_cache_size': 10000,

    # Model inference optimizations
    'model_batch_prediction': True,
    'model_prediction_batch_size': 500,
    'model_cache_predictions': True,

    # Monitoring optimizations
    'monitoring_batch_size': 100,
    'monitoring_flush_interval': 60,
    'monitoring_retention_days': 90,

    # Storage optimizations
    'storage_upload_threads': 8,
    'storage_download_threads': 4,
    'storage_compression': 'gzip'
}
```

### Scaling Configuration

```yaml
# kubernetes/hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rtv-pipeline-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rtv-pipeline-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## ✅ Production Readiness Checklist

### Pre-deployment

- [ ] **Environment Setup**: All dependencies installed and configured
- [ ] **Database Setup**: PostgreSQL and SQLite databases initialized
- [ ] **Model Preparation**: ML models from Part A available and tested
- [ ] **Storage Configuration**: Cloud storage or local storage configured
- [ ] **Monitoring Setup**: Monitoring database and alerts configured
- [ ] **Security Setup**: Encryption keys and access controls in place
- [ ] **Testing**: All tests passing (unit, integration, load)

### Deployment

- [ ] **Docker Images**: Built and tagged for production
- [ ] **Kubernetes Config**: Deployments, services, and ingress configured
- [ ] **Environment Variables**: All production variables set
- [ ] **Secrets Management**: Database passwords and API keys secured
- [ ] **Volume Mounts**: Persistent storage for data, models, monitoring
- [ ] **Resource Limits**: CPU and memory limits appropriate for workload
- [ ] **Health Checks**: Liveness and readiness probes configured

### Post-deployment

- [ ] **Health Verification**: All health checks passing
- [ ] **Feature Engineering**: 50+ features processing correctly
- [ ] **Model Performance**: 97.9% accuracy maintained
- [ ] **Monitoring Active**: Real-time metrics collection working
- [ ] **Alerting Working**: Test alerts sent successfully
- [ ] **Performance Baseline**: Throughput and latency benchmarks established
- [ ] **Scaling Tests**: Auto-scaling working under load
- [ ] **Backup Procedures**: Data and model backup processes verified

### Operational

- [ ] **Monitoring Dashboard**: Team has access to health dashboard
- [ ] **Alert Procedures**: On-call procedures established
- [ ] **Documentation**: Deployment and operational docs complete
- [ ] **Training**: Team trained on monitoring and troubleshooting
- [ ] **Runbooks**: Standard operating procedures documented
- [ ] **Disaster Recovery**: Backup and recovery procedures tested

---

The deployment guide provides comprehensive instructions for deploying the complete ETL pipeline with integrated ML capabilities, monitoring, and production-ready architecture. The pipeline is ready for enterprise-scale deployment and can handle household vulnerability assessment operations in production environments.
