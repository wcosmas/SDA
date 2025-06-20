"""
Part B: Data Engineering Pipeline Configuration
RTV Senior Data Scientist Technical Assessment

This module contains all configuration settings for the ETL pipeline.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # Primary database for processed data
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="rtv_household_data", env="DB_NAME")
    db_user: str = Field(default="rtv_user", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


class StorageConfig(BaseSettings):
    """Cloud storage configuration"""
    
    # Primary storage provider (aws, azure, gcp, local)
    provider: str = Field(default="aws", env="STORAGE_PROVIDER")
    
    # AWS S3 Configuration
    aws_access_key: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: str = Field(default="rtv-household-data", env="S3_BUCKET")
    
    # Azure Configuration
    azure_account_name: str = Field(default="", env="AZURE_ACCOUNT_NAME")
    azure_account_key: str = Field(default="", env="AZURE_ACCOUNT_KEY")
    azure_container: str = Field(default="household-data", env="AZURE_CONTAINER")
    
    # Local storage fallback
    local_storage_path: str = Field(default="./data", env="LOCAL_STORAGE_PATH")


class MLConfig(BaseSettings):
    """Machine Learning pipeline configuration for new dataset structure"""
    
    # Model settings
    model_name: str = "household_vulnerability_predictor_v2"
    model_version: str = "v2.0.0"
    
    # Retraining triggers - updated for higher quality data
    min_new_samples: int = 100  # Higher threshold due to better data quality
    performance_threshold: float = 0.95  # Higher threshold due to excellent model performance
    data_drift_threshold: float = 0.05  # Lower threshold for faster drift detection
    
    # Model storage
    model_registry_path: str = "models/"
    model_artifacts_path: str = "artifacts/"
    
    # Target variable configuration - updated for new assessment thresholds
    target_variable: str = "HHIncome+Consumption+Residues/Day"
    target_thresholds: Dict[str, float] = {
        "on_track": 2.15,
        "at_risk": 1.77,
        "struggling": 1.25
    }
    
    # Key feature categories from the new 75-variable dataset
    feature_categories: Dict[str, List[str]] = {
        "geographic": ["District", "Cluster", "Village"],
        "demographic": ["HouseholdSize"],
        "infrastructure": ["TimeToOPD", "TimeToWater"],
        "agricultural": ["AgricultureLand", "Season1CropsPlanted", "Season2CropsPlanted", "PerennialCropsGrown"],
        "economic": ["VSLA_Profits", "Season1VegetableIncome", "Season2VegatableIncome", "FormalEmployment"],
        "assets": ["VehicleOwner", "BusinessIncome", "Land_owned"],
        "target": ["HHIncome+Consumption+Residues/Day"]
    }
    
    # Essential features for validation and monitoring
    essential_features: List[str] = [
        "HouseholdSize", "AgricultureLand", "VSLA_Profits", "BusinessIncome",
        "VehicleOwner", "District", "HHIncome+Consumption+Residues/Day"
    ]


class PipelineConfig(BaseSettings):
    """Main pipeline configuration"""
    
    # Pipeline execution settings - optimized for larger dataset
    batch_size: int = 500  # Larger batches for 3,897 households
    max_parallel_jobs: int = 8  # More parallel processing
    retry_attempts: int = 3
    timeout_seconds: int = 7200  # Longer timeout for larger data
    
    # Data validation settings - updated for high-quality data
    enable_data_validation: bool = True
    validation_sample_size: float = 0.05  # 5% sample due to better data quality
    completeness_threshold: float = 0.95  # High threshold due to 97.4% completeness
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_retention_days: int = 90
    alert_email: Optional[str] = None
    
    # Performance monitoring thresholds
    expected_accuracy: float = 0.97  # Based on current model performance
    performance_alert_threshold: float = 0.93  # Alert if drops below this
    
    # Scheduling
    schedule_cron: str = "0 2 * * 0"  # Weekly on Sunday at 2 AM
    enable_automatic_retraining: bool = True


class APIConfig(BaseSettings):
    """API service configuration"""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate limiting - updated for field officer usage
    rate_limit_per_minute: int = 200  # Higher limit for field officers
    max_file_size_mb: int = 50  # Smaller files expected with structured data
    max_records_per_upload: int = 1000  # Limit for batch uploads


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = Field(default="logs/pipeline.log", env="LOG_FILE")
    max_file_size_mb: int = 100
    backup_count: int = 5


class ETLConfig:
    """Main ETL configuration class combining all settings"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.storage = StorageConfig()
        self.ml = MLConfig()
        self.pipeline = PipelineConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        # Data paths
        self.paths = {
            "raw_data": "data/raw/",
            "processed_data": "data/processed/",
            "validated_data": "data/validated/",
            "models": "models/",
            "logs": "logs/",
            "temp": "temp/",
            "archives": "archives/",
            "reference_data": "data/reference/"  # For data dictionary
        }
        
        # Dataset-specific settings
        self.dataset_info = {
            "name": "DataScientist_01_Assessment",
            "total_variables": 75,
            "total_households": 3897,
            "completeness_rate": 0.974,
            "districts": 4,
            "villages": 153
        }
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for path in self.paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return (
            f"postgresql://{self.database.db_user}:{self.database.db_password}"
            f"@{self.database.db_host}:{self.database.db_port}/{self.database.db_name}"
        )
    
    def get_storage_config(self) -> Dict:
        """Get storage configuration based on provider"""
        if self.storage.provider == "aws":
            return {
                "provider": "aws",
                "access_key": self.storage.aws_access_key,
                "secret_key": self.storage.aws_secret_key,
                "region": self.storage.aws_region,
                "bucket": self.storage.s3_bucket
            }
        elif self.storage.provider == "azure":
            return {
                "provider": "azure",
                "account_name": self.storage.azure_account_name,
                "account_key": self.storage.azure_account_key,
                "container": self.storage.azure_container
            }
        else:
            return {
                "provider": "local",
                "path": self.storage.local_storage_path
            }
    
    def get_vulnerability_thresholds(self) -> Dict[str, float]:
        """Get vulnerability classification thresholds"""
        return self.ml.target_thresholds
    
    def classify_vulnerability(self, income_per_day: float) -> str:
        """Classify household vulnerability based on income"""
        thresholds = self.ml.target_thresholds
        
        if income_per_day >= thresholds["on_track"]:
            return "On Track"
        elif income_per_day >= thresholds["at_risk"]:
            return "At Risk"
        elif income_per_day >= thresholds["struggling"]:
            return "Struggling"
        else:
            return "Severely Struggling"


# Global configuration instance
config = ETLConfig() 