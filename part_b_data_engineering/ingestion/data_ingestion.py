"""
Part B: Data Ingestion Service
RTV Senior Data Scientist Technical Assessment

Enhanced for DataScientist_01_Assessment.csv structure:
- 75-variable household survey data ingestion
- Real-time field officer data submission
- Batch upload processing for quarterly surveys
- High-quality data validation and processing
- Vulnerability assessment integration
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import json
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from config.pipeline_config import config
from validation.data_validator import DataValidator
from storage.storage_manager import StorageManager

logger = structlog.get_logger(__name__)


class HouseholdSurveyData(BaseModel):
    """Enhanced data model for 75-variable household survey"""
    
    # Essential geographic and identification
    HouseHoldID: str = Field(..., description="Unique household identifier")
    District: str = Field(..., description="District name")
    Cluster: Optional[str] = Field(None, description="Cluster identifier")
    Village: Optional[str] = Field(None, description="Village identifier")
    
    # Demographic data
    HouseholdSize: int = Field(..., ge=1, le=20, description="Number of household members")
    
    # Infrastructure access
    TimeToOPD: Optional[int] = Field(None, ge=0, le=1000, description="Minutes to health facility")
    TimeToWater: Optional[int] = Field(None, ge=0, le=1000, description="Minutes to water source")
    
    # Agricultural data
    AgricultureLand: Optional[float] = Field(None, ge=0, description="Agricultural land in acres")
    Season1CropsPlanted: Optional[int] = Field(None, ge=0, description="Number of crops in season 1")
    Season2CropsPlanted: Optional[int] = Field(None, ge=0, description="Number of crops in season 2")
    PerennialCropsGrown: Optional[int] = Field(None, ge=0, description="Number of perennial crops")
    
    # Economic indicators
    VSLA_Profits: Optional[float] = Field(None, ge=0, description="VSLA profits")
    Season1VegetableIncome: Optional[float] = Field(None, ge=0, description="Season 1 vegetable income")
    Season2VegatableIncome: Optional[float] = Field(None, ge=0, description="Season 2 vegetable income")
    VehicleOwner: Optional[int] = Field(None, ge=0, le=1, description="Vehicle ownership (0/1)")
    BusinessIncome: Optional[float] = Field(None, ge=0, description="Business income")
    FormalEmployment: Optional[int] = Field(None, ge=0, le=1, description="Formal employment (0/1)")
    
    # Target variable - most critical
    target_variable: float = Field(..., alias="HHIncome+Consumption+Residues/Day", 
                                 ge=0, description="Daily household income + consumption + residues")
    
    # Collection metadata
    survey_date: Optional[str] = Field(None, description="Survey collection date")
    field_officer_id: Optional[str] = Field(None, description="Field officer identifier")
    device_id: Optional[str] = Field(None, description="Collection device identifier")
    app_version: Optional[str] = Field(None, description="Mobile app version")
    
    @validator('District')
    def validate_district(cls, v):
        valid_districts = ['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo']
        if v not in valid_districts:
            raise ValueError(f'District must be one of: {valid_districts}')
        return v
    
    @validator('survey_date')
    def validate_survey_date(cls, v):
        if v is not None:
            try:
                pd.to_datetime(v)
            except:
                raise ValueError('Invalid survey_date format')
        return v
    
    class Config:
        allow_population_by_field_name = True


class IngestionResponse(BaseModel):
    """Response model for data ingestion"""
    
    ingestion_id: str
    status: str
    message: str
    records_processed: int
    validation_results: Optional[Dict] = None
    processing_time_seconds: float
    timestamp: str


class BatchIngestionRequest(BaseModel):
    """Request model for batch data ingestion"""
    
    data_source: str = Field(..., description="Source of the data")
    collection_period: str = Field(..., description="Data collection period")
    field_officer_id: Optional[str] = Field(None, description="Field officer responsible")
    notes: Optional[str] = Field(None, description="Additional notes")


class DataIngestionService:
    """Enhanced data ingestion service for 75-variable household surveys"""
    
    def __init__(self):
        self.app = FastAPI(title="RTV Household Survey Data Ingestion API v2.0")
        self.validator = DataValidator()
        self.storage_manager = StorageManager()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes for data ingestion"""
        
        @self.app.post("/api/v2/ingest/household", response_model=IngestionResponse)
        async def ingest_household_data(
            data: HouseholdSurveyData,
            background_tasks: BackgroundTasks
        ):
            """Ingest individual household survey data"""
            return await self.process_individual_record(data, background_tasks)
        
        @self.app.post("/api/v2/ingest/batch", response_model=IngestionResponse)
        async def ingest_batch_data(
            file: UploadFile = File(...),
            request: BatchIngestionRequest = None,
            background_tasks: BackgroundTasks = None
        ):
            """Ingest batch household survey data from CSV file"""
            return await self.process_batch_upload(file, request, background_tasks)
        
        @self.app.get("/api/v2/health")
        async def health_check():
            """API health check endpoint"""
            return {
                "status": "healthy",
                "version": "2.0.0",
                "dataset_version": "DataScientist_01_Assessment",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "supported_variables": 75,
                "expected_completeness": "97.4%"
            }
        
        @self.app.get("/api/v2/stats")
        async def get_ingestion_stats():
            """Get ingestion statistics"""
            return await self.get_processing_statistics()
    
    async def process_individual_record(
        self, 
        data: HouseholdSurveyData, 
        background_tasks: BackgroundTasks
    ) -> IngestionResponse:
        """Process individual household survey record with enhanced validation"""
        
        start_time = datetime.now(timezone.utc)
        ingestion_id = str(uuid.uuid4())
        
        try:
            logger.info("Processing individual household record", 
                       ingestion_id=ingestion_id,
                       household_id=data.HouseHoldID)
            
            # Convert to dictionary for processing
            record_dict = data.dict(by_alias=True)
            
            # Enhanced validation using the updated validator
            validation_result = self.validator.validate_survey_data(record_dict)
            
            if not validation_result.is_valid:
                logger.warning("Validation failed for household record",
                             ingestion_id=ingestion_id,
                             errors=validation_result.errors)
                
                return IngestionResponse(
                    ingestion_id=ingestion_id,
                    status="validation_failed",
                    message=f"Validation failed: {'; '.join(validation_result.errors)}",
                    records_processed=0,
                    validation_results=validation_result.__dict__,
                    processing_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            # Add vulnerability classification
            record_dict['vulnerability_class'] = config.classify_vulnerability(
                record_dict['HHIncome+Consumption+Residues/Day']
            )
            record_dict['is_vulnerable'] = record_dict['vulnerability_class'] in ['Struggling', 'Severely Struggling']
            
            # Add processing metadata
            record_dict['ingestion_id'] = ingestion_id
            record_dict['ingestion_timestamp'] = start_time.isoformat()
            record_dict['processing_version'] = '2.0.0'
            
            # Store the processed record
            background_tasks.add_task(
                self.store_individual_record,
                record_dict,
                ingestion_id
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info("Individual record processed successfully",
                       ingestion_id=ingestion_id,
                       processing_time=processing_time)
            
            return IngestionResponse(
                ingestion_id=ingestion_id,
                status="success",
                message="Household record processed successfully",
                records_processed=1,
                validation_results={
                    "is_valid": True,
                    "warnings": validation_result.warnings,
                    "completeness_rate": validation_result.metrics.get('completeness_rate', 0)
                },
                processing_time_seconds=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error("Failed to process individual record",
                        ingestion_id=ingestion_id,
                        error=str(e))
            
            return IngestionResponse(
                ingestion_id=ingestion_id,
                status="error",
                message=f"Processing failed: {str(e)}",
                records_processed=0,
                validation_results=None,
                processing_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def process_batch_upload(
        self,
        file: UploadFile,
        request: BatchIngestionRequest,
        background_tasks: BackgroundTasks
    ) -> IngestionResponse:
        """Process batch upload of household survey data"""
        
        start_time = datetime.now(timezone.utc)
        ingestion_id = str(uuid.uuid4())
        
        try:
            logger.info("Processing batch upload",
                       ingestion_id=ingestion_id,
                       filename=file.filename,
                       content_type=file.content_type)
            
            # Read uploaded file
            content = await file.read()
            
            # Support multiple file formats
            if file.filename.endswith('.csv'):
                df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(pd.io.common.BytesIO(content))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Please upload CSV or Excel files."
                )
            
            logger.info("File loaded successfully",
                       ingestion_id=ingestion_id,
                       records=len(df),
                       columns=len(df.columns))
            
            # Enhanced batch validation
            validation_result = self.validator.validate_batch_data(df)
            
            if not validation_result.is_valid:
                logger.warning("Batch validation failed",
                             ingestion_id=ingestion_id,
                             errors=validation_result.errors,
                             failed_records=validation_result.failed_records)
            
            # Apply vulnerability classification to all records
            if 'HHIncome+Consumption+Residues/Day' in df.columns:
                df['vulnerability_class'] = df['HHIncome+Consumption+Residues/Day'].apply(
                    config.classify_vulnerability
                )
                df['is_vulnerable'] = df['vulnerability_class'].isin(['Struggling', 'Severely Struggling'])
            
            # Add batch processing metadata
            df['ingestion_id'] = ingestion_id
            df['ingestion_timestamp'] = start_time.isoformat()
            df['processing_version'] = '2.0.0'
            df['data_source'] = request.data_source if request else 'batch_upload'
            
            # Process in background
            background_tasks.add_task(
                self.store_batch_data,
                df,
                ingestion_id,
                validation_result
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Determine success status
            success_rate = (len(df) - validation_result.failed_records) / len(df)
            status = "success" if success_rate > 0.95 else "partial_success"
            
            logger.info("Batch processing completed",
                       ingestion_id=ingestion_id,
                       total_records=len(df),
                       failed_records=validation_result.failed_records,
                       success_rate=success_rate,
                       processing_time=processing_time)
            
            return IngestionResponse(
                ingestion_id=ingestion_id,
                status=status,
                message=f"Batch processing completed. {len(df) - validation_result.failed_records}/{len(df)} records processed successfully.",
                records_processed=len(df) - validation_result.failed_records,
                validation_results={
                    "total_records": validation_result.validated_records,
                    "failed_records": validation_result.failed_records,
                    "success_rate": success_rate,
                    "data_completeness": validation_result.metrics.get('completeness_rate', 0),
                    "vulnerability_distribution": df['vulnerability_class'].value_counts().to_dict() if 'vulnerability_class' in df.columns else {}
                },
                processing_time_seconds=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error("Batch processing failed",
                        ingestion_id=ingestion_id,
                        error=str(e))
            
            return IngestionResponse(
                ingestion_id=ingestion_id,
                status="error",
                message=f"Batch processing failed: {str(e)}",
                records_processed=0,
                validation_results=None,
                processing_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    async def store_individual_record(self, record_dict: Dict, ingestion_id: str):
        """Store individual record with enhanced metadata"""
        try:
            # Create filename with vulnerability info
            vulnerability_class = record_dict.get('vulnerability_class', 'unknown')
            filename = f"household_{record_dict['HouseHoldID']}_{vulnerability_class}_{ingestion_id}.json"
            file_path = f"individual_records/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            
            await self.storage_manager.store_data(
                data=record_dict,
                file_path=file_path,
                metadata={
                    'type': 'individual_household_record',
                    'ingestion_id': ingestion_id,
                    'household_id': record_dict['HouseHoldID'],
                    'vulnerability_class': vulnerability_class,
                    'district': record_dict.get('District'),
                    'processing_version': '2.0.0'
                }
            )
            
            logger.info("Individual record stored successfully",
                       ingestion_id=ingestion_id,
                       file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to store individual record",
                        ingestion_id=ingestion_id,
                        error=str(e))
    
    async def store_batch_data(self, df: pd.DataFrame, ingestion_id: str, validation_result):
        """Store batch data with enhanced organization"""
        try:
            # Create organized storage structure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_path = f"batch_data/{datetime.now().strftime('%Y/%m')}"
            
            # Store main dataset
            main_file_path = f"{base_path}/household_survey_{timestamp}_{ingestion_id}.csv"
            await self.storage_manager.store_data(
                data=df.to_csv(index=False),
                file_path=main_file_path,
                metadata={
                    'type': 'batch_household_survey',
                    'ingestion_id': ingestion_id,
                    'total_records': len(df),
                    'processing_version': '2.0.0',
                    'data_completeness': validation_result.metrics.get('completeness_rate', 0)
                }
            )
            
            # Store validation report
            validation_file_path = f"{base_path}/validation_report_{timestamp}_{ingestion_id}.json"
            await self.storage_manager.store_data(
                data=json.dumps(validation_result.__dict__, default=str, indent=2),
                file_path=validation_file_path,
                metadata={
                    'type': 'validation_report',
                    'ingestion_id': ingestion_id,
                    'processing_version': '2.0.0'
                }
            )
            
            # Store vulnerability analysis
            if 'vulnerability_class' in df.columns:
                vulnerability_summary = {
                    'total_households': len(df),
                    'vulnerability_distribution': df['vulnerability_class'].value_counts().to_dict(),
                    'vulnerability_rate': df['is_vulnerable'].mean(),
                    'district_vulnerability': df.groupby('District')['is_vulnerable'].agg(['count', 'mean']).to_dict(),
                    'processing_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                vuln_file_path = f"{base_path}/vulnerability_analysis_{timestamp}_{ingestion_id}.json"
                await self.storage_manager.store_data(
                    data=json.dumps(vulnerability_summary, default=str, indent=2),
                    file_path=vuln_file_path,
                    metadata={
                        'type': 'vulnerability_analysis',
                        'ingestion_id': ingestion_id,
                        'processing_version': '2.0.0'
                    }
                )
            
            logger.info("Batch data stored successfully",
                       ingestion_id=ingestion_id,
                       main_file=main_file_path,
                       records_stored=len(df))
            
        except Exception as e:
            logger.error("Failed to store batch data",
                        ingestion_id=ingestion_id,
                        error=str(e))
    
    async def get_processing_statistics(self) -> Dict:
        """Get enhanced processing statistics"""
        try:
            # In a real implementation, this would query the storage system
            # For demo purposes, we'll return simulated statistics
            
            stats = {
                'api_version': '2.0.0',
                'dataset_version': 'DataScientist_01_Assessment',
                'total_variables_supported': 75,
                'processing_statistics': {
                    'total_records_processed': 1247,  # Simulated
                    'records_last_24h': 89,
                    'records_last_week': 456,
                    'average_processing_time_ms': 145,
                    'validation_success_rate': 98.7,
                    'data_completeness_average': 97.4
                },
                'vulnerability_statistics': {
                    'total_households_assessed': 1247,
                    'vulnerable_households': 534,
                    'vulnerability_rate': 42.8,
                    'critical_risk_households': 187,
                    'high_risk_households': 347
                },
                'geographic_coverage': {
                    'districts_covered': 4,
                    'villages_covered': 145,
                    'field_officers_active': 28
                },
                'data_quality_metrics': {
                    'completeness_threshold_met': True,
                    'validation_threshold_met': True,
                    'drift_detected': False,
                    'last_model_update': '2024-01-15T10:30:00Z'
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get processing statistics", error=str(e))
            return {
                'error': 'Unable to retrieve statistics',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


class BatchIngestionProcessor:
    """Enhanced batch ingestion processor for quarterly data updates"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.storage_manager = StorageManager()
    
    async def process_quarterly_data(self, file_path: str) -> Dict:
        """Process quarterly household survey data"""
        
        start_time = datetime.now(timezone.utc)
        processing_id = str(uuid.uuid4())
        
        try:
            logger.info("Starting quarterly data processing",
                       processing_id=processing_id,
                       file_path=file_path)
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Enhanced validation for quarterly data
            validation_result = self.validator.validate_batch_data(df)
            
            # Apply vulnerability classification
            if 'HHIncome+Consumption+Residues/Day' in df.columns:
                df['vulnerability_class'] = df['HHIncome+Consumption+Residues/Day'].apply(
                    config.classify_vulnerability
                )
                df['is_vulnerable'] = df['vulnerability_class'].isin(['Struggling', 'Severely Struggling'])
                
                # Generate intervention flags
                df['intervention_priority'] = df.apply(self._calculate_intervention_priority, axis=1)
            
            # Add quarterly processing metadata
            df['processing_id'] = processing_id
            df['processing_timestamp'] = start_time.isoformat()
            df['processing_type'] = 'quarterly_assessment'
            df['processing_version'] = '2.0.0'
            
            # Store processed quarterly data
            timestamp = datetime.now().strftime('%Y_Q%m')
            output_file = f"quarterly_data/processed_household_survey_{timestamp}_{processing_id}.csv"
            
            await self.storage_manager.store_data(
                data=df.to_csv(index=False),
                file_path=output_file,
                metadata={
                    'type': 'quarterly_household_assessment',
                    'processing_id': processing_id,
                    'total_records': len(df),
                    'vulnerability_rate': df['is_vulnerable'].mean() if 'is_vulnerable' in df.columns else None,
                    'processing_version': '2.0.0'
                }
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = {
                'processing_id': processing_id,
                'status': 'success',
                'records_processed': len(df),
                'validation_results': validation_result.__dict__,
                'vulnerability_assessment': {
                    'total_households': len(df),
                    'vulnerable_count': df['is_vulnerable'].sum() if 'is_vulnerable' in df.columns else 0,
                    'vulnerability_rate': df['is_vulnerable'].mean() if 'is_vulnerable' in df.columns else 0,
                    'high_priority_interventions': len(df[df['intervention_priority'] == 'high']) if 'intervention_priority' in df.columns else 0
                } if 'is_vulnerable' in df.columns else {},
                'processing_time_seconds': processing_time,
                'output_file': output_file,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Quarterly data processing completed",
                       processing_id=processing_id,
                       records=len(df),
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Quarterly data processing failed",
                        processing_id=processing_id,
                        error=str(e))
            
            return {
                'processing_id': processing_id,
                'status': 'error',
                'message': str(e),
                'processing_time_seconds': (datetime.now(timezone.utc) - start_time).total_seconds(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_intervention_priority(self, row) -> str:
        """Calculate intervention priority based on vulnerability and other factors"""
        try:
            vulnerability_class = row.get('vulnerability_class', 'Unknown')
            household_size = row.get('HouseholdSize', 0)
            
            if vulnerability_class == 'Severely Struggling':
                return 'critical'
            elif vulnerability_class == 'Struggling' and household_size > 8:
                return 'high'
            elif vulnerability_class in ['Struggling', 'At Risk']:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'unknown'


# Initialize global services
data_ingestion_service = DataIngestionService()
batch_processor = BatchIngestionProcessor()

# Export FastAPI app for deployment
app = data_ingestion_service.app 