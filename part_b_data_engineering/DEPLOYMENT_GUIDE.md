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
# Set storage provider, database credentials, etc.
```

### 3. Database Setup

```sql
-- For PostgreSQL
CREATE DATABASE rtv_household_data;
CREATE USER pipeline_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rtv_household_data TO pipeline_user;
```

### 4. Run Pipeline Demo

```bash
# Run comprehensive demonstration
python demo_pipeline.py

# Expected output: 100% success rate with generated predictions
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
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rtv_data
      - STORAGE_PROVIDER=aws
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=rtv_household_data
      - POSTGRES_USER=pipeline_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  celery-worker:
    build: .
    command: celery -A pipeline.etl_orchestrator worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/rtv_data
    depends_on:
      - postgres
      - redis

  scheduler:
    build: .
    command: python -c "from pipeline.etl_orchestrator import create_etl_orchestrator; import asyncio; asyncio.run(create_etl_orchestrator().start_pipeline())"
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
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

---

## 🔧 Configuration Guide

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/rtv_data
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Storage Configuration
STORAGE_PROVIDER=aws  # aws, azure, or local
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_BUCKET_NAME=rtv-household-data
AWS_REGION=us-east-1

# Security Configuration
SECRET_KEY=your-256-bit-secret-key
ENCRYPTION_KEY=your-fernet-encryption-key
JWT_EXPIRATION_HOURS=24

# Processing Configuration
MAX_WORKERS=4
BATCH_SIZE=1000
VALIDATION_STRICT_MODE=true

# Monitoring Configuration
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
ALERT_EMAIL=alerts@organization.com
```

### Cloud Provider Setup

#### AWS Configuration

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: [your_key]
# AWS Secret Access Key: [your_secret]
# Default region: us-east-1
# Default output format: json

# Create S3 bucket
aws s3 mb s3://rtv-household-data

# Set bucket policy for encryption
aws s3api put-bucket-encryption \
  --bucket rtv-household-data \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
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
  --name rtvhouseholddata \
  --resource-group rtv-pipeline \
  --location eastus \
  --sku Standard_LRS

# Get storage keys
az storage account keys list \
  --account-name rtvhouseholddata \
  --resource-group rtv-pipeline
```

---

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "rtv-pipeline"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "postgres"
    static_configs:
      - targets: ["localhost:9187"]

rule_files:
  - "pipeline_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RTV Pipeline Monitoring",
    "panels": [
      {
        "title": "Pipeline Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(pipeline_success_total[5m]) / rate(pipeline_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "Processing Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(records_processed_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# pipeline_alerts.yml
groups:
  - name: pipeline_alerts
    rules:
      - alert: PipelineFailureRate
        expr: rate(pipeline_failures_total[5m]) / rate(pipeline_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline failure rate is {{ $value | humanizePercentage }}"

      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "95th percentile latency is {{ $value }}s"
```

---

## 🧪 Testing & Validation

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Integration Testing

```bash
# Test database connection
python -c "
from storage.storage_manager import StorageManager
manager = StorageManager()
print('✓ Database connection successful')
"

# Test storage backend
python -c "
from storage.storage_manager import StorageManager
import pandas as pd
manager = StorageManager()
df = pd.DataFrame({'test': [1, 2, 3]})
path = manager.save_data(df, 'test_data', 'raw')
print(f'✓ Storage test successful: {path}')
"

# Test API endpoints
curl -X POST http://localhost:8000/api/v1/upload/survey-data \
  -H "Content-Type: application/json" \
  -d '{"household_id": "test_001", "region": 1, "ppi_score": 50}'
```

### Load Testing

```python
# locustfile.py
from locust import HttpUser, task, between

class PipelineUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def submit_survey_data(self):
        self.client.post("/api/v1/upload/survey-data", json={
            "household_id": f"test_{self.user_id}",
            "region": 1,
            "ppi_score": 50,
            "household_size": 5
        })

    @task
    def check_status(self):
        self.client.get("/api/v1/status/test_ingestion_id")
```

---

## 🔒 Security Checklist

### Pre-deployment Security

- [ ] **Encryption Keys**: Generate and securely store Fernet encryption keys
- [ ] **Database Credentials**: Use strong passwords and connection pooling
- [ ] **API Authentication**: Implement JWT token authentication
- [ ] **Network Security**: Configure firewalls and VPCs
- [ ] **SSL/TLS**: Enable HTTPS for all endpoints
- [ ] **Input Validation**: Verify all Pydantic models are properly configured
- [ ] **Rate Limiting**: Configure API rate limits
- [ ] **Audit Logging**: Enable comprehensive audit trails

### Post-deployment Security

- [ ] **Regular Updates**: Keep dependencies updated
- [ ] **Security Scanning**: Run vulnerability scans
- [ ] **Access Reviews**: Regular access control audits
- [ ] **Backup Encryption**: Verify backup encryption
- [ ] **Incident Response**: Test incident response procedures

---

## 🚨 Troubleshooting

### Common Issues

#### Database Connection Errors

```bash
# Check database status
pg_isready -h localhost -p 5432

# Test connection
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@localhost:5432/rtv_data')
print('✓ Database connection successful')
conn.close()
"
```

#### Storage Backend Issues

```bash
# Test AWS connection
aws s3 ls s3://rtv-household-data

# Test Azure connection
az storage blob list --account-name rtvhouseholddata --container-name data
```

#### Memory Issues

```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / (1024**3):.1f}GB')
"

# Optimize batch sizes in configuration
# Reduce MAX_WORKERS if memory constrained
```

#### API Performance Issues

```bash
# Check API health
curl http://localhost:8000/health

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/status/test
```

### Performance Tuning

#### Database Optimization

```sql
-- Create indexes for common queries
CREATE INDEX idx_household_region ON household_data(region);
CREATE INDEX idx_processing_status ON ingestion_log(processing_status);
CREATE INDEX idx_created_at ON ingestion_log(created_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM household_data WHERE region = 1;
```

#### Storage Optimization

```python
# Use Parquet for better compression
df.to_parquet('data.parquet', compression='snappy')

# Implement data partitioning by date
partition_path = f"year={year}/month={month}/day={day}/data.parquet"
```

---

## 📈 Scaling Guidelines

### Horizontal Scaling

- **API Instances**: Scale to 3-5 instances behind load balancer
- **Worker Processes**: Scale Celery workers based on queue length
- **Database**: Use read replicas for query scaling
- **Storage**: Implement data partitioning strategies

### Vertical Scaling

- **CPU**: 4-8 cores for processing-intensive workloads
- **Memory**: 16-32GB for large batch processing
- **Storage**: NVMe SSDs for database performance
- **Network**: High-bandwidth connections for cloud storage

### Auto-scaling Configuration

```yaml
# kubernetes/hpa.yaml
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

## 🔄 Maintenance Procedures

### Daily Maintenance

```bash
#!/bin/bash
# daily_maintenance.sh

# Check pipeline health
python -c "from pipeline.etl_orchestrator import HealthChecker; HealthChecker().check_all()"

# Clean temporary files
find temp/ -name "*.tmp" -mtime +1 -delete

# Backup configuration
cp config/pipeline_config.py backups/config_$(date +%Y%m%d).py

# Check disk space
df -h | grep -E "(80%|90%|100%)" && echo "WARNING: Low disk space"
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

# Archive old logs
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# Update model performance metrics
python -c "from pipeline.etl_orchestrator import ModelMonitor; ModelMonitor().generate_weekly_report()"

# Database maintenance
psql -d rtv_household_data -c "VACUUM ANALYZE;"

# Security scan
pip-audit --desc --output audit_$(date +%Y%m%d).json
```

### Monthly Maintenance

```bash
#!/bin/bash
# monthly_maintenance.sh

# Full database backup
pg_dump rtv_household_data > backups/db_backup_$(date +%Y%m%d).sql

# Archive processed data
python -c "from storage.storage_manager import DataArchiver; DataArchiver().archive_monthly_data()"

# Update dependencies
pip list --outdated > outdated_packages_$(date +%Y%m%d).txt

# Capacity planning review
python -c "from monitoring.capacity_planner import CapacityPlanner; CapacityPlanner().generate_monthly_report()"
```

---

## 📞 Support & Contacts

### Support Tiers

#### Tier 1: Self-Service

- **Documentation**: This deployment guide and architecture docs
- **Health Checks**: Built-in monitoring dashboards
- **Common Issues**: Troubleshooting section above

#### Tier 2: Remote Support

- **Response Time**: 4 hours during business hours
- **Coverage**: Configuration issues, performance tuning
- **Contact**: pipeline-support@organization.com

#### Tier 3: Critical Issues

- **Response Time**: 1 hour, 24/7
- **Coverage**: Production outages, data loss incidents
- **Contact**: critical-support@organization.com

### Emergency Procedures

```bash
# Pipeline emergency stop
python -c "from pipeline.etl_orchestrator import EmergencyStop; EmergencyStop().halt_all_processing()"

# Rollback to previous model version
python -c "from pipeline.model_manager import ModelManager; ModelManager().rollback_to_previous()"

# Database emergency readonly mode
psql -d rtv_household_data -c "ALTER DATABASE rtv_household_data SET default_transaction_read_only = on;"
```

---

## ✅ Deployment Verification

### Post-deployment Checklist

- [ ] **Services Running**: All containers/services are healthy
- [ ] **Database Connected**: Connection pool is active
- [ ] **Storage Accessible**: Can read/write to configured storage
- [ ] **API Responsive**: All endpoints return expected responses
- [ ] **Authentication Working**: JWT tokens are validated
- [ ] **Monitoring Active**: Metrics are being collected
- [ ] **Alerts Configured**: Test alerts are sent successfully
- [ ] **Backups Configured**: Automated backups are running
- [ ] **SSL Certificates**: HTTPS is working properly
- [ ] **Performance Baseline**: Response times meet SLA

### Validation Tests

```bash
# Run comprehensive validation
python scripts/validate_deployment.py

# Expected output:
# ✓ Database connection: PASS
# ✓ Storage backend: PASS
# ✓ API endpoints: PASS
# ✓ Authentication: PASS
# ✓ Monitoring: PASS
# ✓ Model loading: PASS
# ✓ End-to-end pipeline: PASS
#
# Deployment Status: READY FOR PRODUCTION
```

---

**Deployment Guide Complete**  
**Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Ready for Production Deployment

For technical support or questions about this deployment guide, contact the development team or refer to the troubleshooting section above.
