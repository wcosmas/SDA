# Part B: Data Engineering Pipeline

## RTV Senior Data Scientist Technical Assessment

This directory contains the complete ETL pipeline solution for automated household survey data processing, validation, and model retraining.

## 🏗️ Architecture Overview

The ETL pipeline is designed as a modular, scalable system that handles:

- **Data Ingestion**: Multi-source data collection from field devices
- **Data Validation**: Comprehensive quality checks and business rule validation
- **Data Transformation**: Feature engineering and data processing
- **Model Management**: Automated retraining and prediction generation
- **Monitoring**: Real-time pipeline health and performance tracking

## 📁 Directory Structure

```
part_b_data_engineering/
├── config/
│   └── pipeline_config.py          # Configuration management
├── ingestion/
│   └── data_ingestion.py           # Data ingestion service
├── validation/
│   └── data_validator.py           # Data quality validation
├── storage/
│   └── storage_manager.py          # Multi-cloud storage manager
├── pipeline/
│   └── etl_orchestrator.py         # Main pipeline orchestrator
├── monitoring/
│   └── [monitoring components]     # Pipeline monitoring
├── demo_pipeline.py                # Demonstration script
├── requirements.txt                # Python dependencies
├── ARCHITECTURE_DOCUMENTATION.md  # Detailed architecture docs
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd part_b_data_engineering
pip install -r requirements.txt
```

### 2. Run the Demonstration

The easiest way to understand the pipeline is to run the demonstration:

```bash
python demo_pipeline.py
```

This will:

- Create sample household survey data
- Demonstrate all pipeline components
- Show data validation, transformation, and prediction generation
- Generate a comprehensive report

### 3. Configure for Production

Copy and customize the configuration:

```python
# In config/pipeline_config.py
config.storage.provider = "aws"  # or "azure" or "local"
config.database.db_host = "your-database-host"
config.pipeline.alert_email = "admin@yourorganization.org"
```

### 4. Start the Pipeline

```python
from pipeline.etl_orchestrator import create_etl_orchestrator

orchestrator = create_etl_orchestrator()
await orchestrator.start_pipeline()
```

## 🔧 Key Components

### Data Ingestion Service

**Purpose**: Handles data collection from multiple sources
**Features**:

- FastAPI endpoints for real-time data submission
- Batch file upload support (CSV, Excel, JSON, Parquet)
- Background task processing with Celery
- Unique ingestion ID tracking

**Usage**:

```python
from ingestion.data_ingestion import DataIngestionService

service = DataIngestionService()
# API automatically starts on port 8000
```

**API Endpoints**:

- `POST /api/v1/upload/survey-data` - Submit individual survey
- `POST /api/v1/upload/file` - Upload batch file
- `GET /api/v1/status/{ingestion_id}` - Check processing status

### Data Validation

**Purpose**: Ensures data quality and consistency
**Features**:

- Schema validation (required fields, data types)
- Business rule validation (cross-field dependencies)
- Statistical anomaly detection
- Great Expectations integration

**Usage**:

```python
from validation.data_validator import DataValidator

validator = DataValidator()
result = validator.validate_batch_data(dataframe)

if result.is_valid:
    print(f"✓ {result.validated_records} records validated")
else:
    print(f"✗ {result.failed_records} records failed validation")
    print(f"Errors: {result.errors}")
```

### Storage Manager

**Purpose**: Unified storage across multiple cloud providers
**Features**:

- Multi-cloud support (AWS S3, Azure Blob, Local)
- Data encryption for sensitive fields
- Automatic metadata generation
- Data archiving and lifecycle management

**Usage**:

```python
from storage.storage_manager import StorageManager

storage = StorageManager()

# Store data
await storage.store_data(dataframe, "processed/survey_data.parquet")

# Retrieve data
data = await storage.retrieve_data("processed/survey_data.parquet")

# List files
files = await storage.list_files("processed/")
```

### ETL Orchestrator

**Purpose**: Coordinates the entire pipeline
**Features**:

- Scheduled pipeline execution (cron-based)
- Job dependencies and parallel execution
- Retry logic with exponential backoff
- Comprehensive monitoring and alerting

**Usage**:

```python
from pipeline.etl_orchestrator import ETLOrchestrator

orchestrator = ETLOrchestrator()
await orchestrator.start_pipeline()

# Get pipeline status
status = orchestrator.get_pipeline_status()
print(f"Last run: {status['last_run']}")
print(f"Next run: {status['next_scheduled_run']}")
```

## 📊 Data Flow

### 1. Data Collection

```
Field Devices → API Gateway → Validation → Temporary Storage
```

### 2. Data Processing

```
Raw Data → Quality Checks → Feature Engineering → Processed Data
```

### 3. Model Pipeline

```
Processed Data → Training Trigger → Model Update → Predictions
```

### 4. Storage Hierarchy

```
/raw         # Unprocessed ingested data
/validated   # Quality-checked data
/processed   # Feature-engineered data
/predictions # Model outputs
/archive     # Long-term storage
```

## 🔒 Security Features

### Data Protection

- **Encryption**: Sensitive fields encrypted with Fernet
- **Access Control**: JWT-based authentication
- **Audit Logging**: Complete data lineage tracking
- **Data Masking**: PII protection in non-production

### Network Security

- **TLS Encryption**: All data in transit protected
- **Rate Limiting**: API abuse protection
- **IP Whitelisting**: Restricted access
- **VPC Isolation**: Cloud resources in private networks

## 📈 Monitoring & Alerting

### Pipeline Metrics

- **Performance**: Execution times and throughput
- **Quality**: Data validation success rates
- **Health**: Component status and availability
- **Business**: Processing volumes and predictions

### Alerting

- **Email Notifications**: Pipeline failures and warnings
- **Threshold Alerts**: Performance and quality issues
- **Health Checks**: Component availability monitoring

## 🔄 Model Retraining

### Automatic Triggers

- **Data Volume**: Minimum new samples threshold (50+)
- **Performance**: Model accuracy degradation
- **Time-based**: Scheduled periodic updates
- **Manual**: API-triggered retraining

### Retraining Process

1. Detect trigger conditions
2. Prepare training dataset
3. Train new model version
4. Validate performance
5. Deploy if improved
6. Archive previous version

## 🌍 Deployment Options

### Development

- **Local Storage**: File system for testing
- **SQLite Database**: Lightweight metadata storage
- **Single Process**: All components in one process

### Staging

- **Cloud Storage**: AWS S3 or Azure Blob
- **PostgreSQL**: Dedicated database instance
- **Container Deployment**: Docker/Kubernetes

### Production

- **Multi-cloud**: Primary and backup storage
- **High Availability**: Load balancers and replicas
- **Auto-scaling**: Dynamic resource allocation
- **Monitoring**: Full observability stack

## 📋 Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rtv_household_data
DB_USER=rtv_user
DB_PASSWORD=your_password

# Storage Configuration
STORAGE_PROVIDER=aws
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=rtv-household-data

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your_secret_key

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log
```

### Pipeline Configuration

```python
# Modify config/pipeline_config.py
config.pipeline.schedule_cron = "0 2 * * 0"  # Weekly on Sunday
config.ml.min_new_samples = 50
config.ml.performance_threshold = 0.85
config.pipeline.enable_monitoring = True
```

## 🧪 Testing

### Run Demonstration

```bash
python demo_pipeline.py
```

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Load Testing

```bash
locust -f tests/load_test.py
```

## 📚 Additional Documentation

- **[Architecture Documentation](ARCHITECTURE_DOCUMENTATION.md)**: Detailed system design
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs (when running)
- **[Configuration Guide](config/README.md)**: Configuration options
- **[Deployment Guide](deployment/README.md)**: Production deployment

## 🎯 Business Benefits

### For Field Officers

- **Real-time Validation**: Immediate feedback on data quality
- **Offline Support**: Queue-based processing for poor connectivity
- **Simple Integration**: RESTful APIs for mobile apps

### For Data Teams

- **Automated Processing**: Reduced manual intervention
- **Quality Assurance**: Comprehensive validation
- **Version Control**: Complete data lineage

### For Program Managers

- **Faster Insights**: Automated quarterly processing
- **Quality Reporting**: Data quality dashboards
- **Cost Optimization**: Efficient resource utilization

## 🔧 Troubleshooting

### Common Issues

**Pipeline Not Starting**

```bash
# Check configuration
python -c "from config.pipeline_config import config; print(config.get_database_url())"

# Check dependencies
pip install -r requirements.txt

# Check logs
tail -f logs/pipeline.log
```

**Data Validation Failing**

```bash
# Run validation test
python -c "
from validation.data_validator import DataValidator
import pandas as pd
validator = DataValidator()
df = pd.read_csv('sample_data.csv')
result = validator.validate_batch_data(df)
print(result.errors)
"
```

**Storage Connection Issues**

```bash
# Test storage connection
python -c "
from storage.storage_manager import StorageManager
storage = StorageManager()
print(storage.storage_config)
"
```

### Performance Optimization

**High Memory Usage**

- Increase chunk size for batch processing
- Use streaming for large files
- Configure garbage collection

**Slow Processing**

- Enable parallel processing
- Optimize database queries
- Use connection pooling

**Storage Costs**

- Configure data lifecycle policies
- Use compression (Parquet format)
- Archive old data regularly

## 📞 Support

For technical support or questions about the ETL pipeline:

1. **Check Documentation**: Review architecture docs and README files
2. **Run Diagnostics**: Use the demonstration script to test components
3. **Check Logs**: Review pipeline logs for error details
4. **Contact Team**: Reach out to the data engineering team

## 🚀 Next Steps

### Phase 1: Basic Deployment

- [ ] Deploy to staging environment
- [ ] Configure cloud storage
- [ ] Set up monitoring dashboards
- [ ] Test with real data

### Phase 2: Production Ready

- [ ] High availability setup
- [ ] Automated testing pipeline
- [ ] Performance optimization
- [ ] Security hardening

### Phase 3: Advanced Features

- [ ] Real-time streaming
- [ ] Advanced ML models
- [ ] Graph analytics
- [ ] Mobile SDK

---

**Note**: This ETL pipeline is designed for the RTV Senior Data Scientist Technical Assessment and demonstrates production-ready data engineering practices for household vulnerability assessment in last-mile communities.
