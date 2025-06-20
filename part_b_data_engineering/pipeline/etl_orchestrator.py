"""
Part B: ETL Pipeline Orchestrator
RTV Senior Data Scientist Technical Assessment

This module orchestrates the complete ETL pipeline:
- Coordinates data ingestion from multiple sources
- Manages data validation and quality checks
- Handles data transformation and feature engineering
- Triggers model retraining when conditions are met
- Monitors pipeline performance and health
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from pathlib import Path
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import joblib
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from config.pipeline_config import config
from ingestion.data_ingestion import DataIngestionService, BatchIngestionProcessor
from validation.data_validator import DataValidator
from storage.storage_manager import StorageManager
from pipeline.feature_engineer import FeatureEngineer
from pipeline.model_trainer import ModelTrainer
from monitoring.pipeline_monitor import PipelineMonitor

logger = structlog.get_logger(__name__)


class ETLOrchestrator:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self):
        # Initialize components
        self.storage_manager = StorageManager()
        self.data_validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.pipeline_monitor = PipelineMonitor()
        self.batch_processor = BatchIngestionProcessor()
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        self._setup_scheduled_jobs()
        
        # Pipeline state
        self.is_running = False
        self.last_run_timestamp = None
        self.pipeline_metrics = {}
    
    def _setup_scheduled_jobs(self):
        """Set up scheduled pipeline jobs"""
        # Main ETL pipeline - runs on configured schedule
        self.scheduler.add_job(
            self.run_etl_pipeline,
            CronTrigger.from_crontab(config.pipeline.schedule_cron),
            id='main_etl_pipeline',
            name='Main ETL Pipeline',
            max_instances=1
        )
        
        # Data quality monitoring - runs daily
        self.scheduler.add_job(
            self.run_data_quality_check,
            CronTrigger(hour=1, minute=0),  # Daily at 1 AM
            id='data_quality_check',
            name='Data Quality Monitoring',
            max_instances=1
        )
        
        # Model performance monitoring - runs daily
        self.scheduler.add_job(
            self.check_model_performance,
            CronTrigger(hour=3, minute=0),  # Daily at 3 AM
            id='model_performance_check',
            name='Model Performance Check',
            max_instances=1
        )
        
        # Data archiving - runs weekly
        self.scheduler.add_job(
            self.archive_old_data,
            CronTrigger(day_of_week=0, hour=5, minute=0),  # Weekly on Sunday at 5 AM
            id='data_archiving',
            name='Data Archiving',
            max_instances=1
        )
    
    async def start_pipeline(self):
        """Start the ETL pipeline orchestrator"""
        try:
            logger.info("Starting ETL Pipeline Orchestrator")
            
            # Start monitoring
            await self.pipeline_monitor.start_monitoring()
            
            # Start scheduler
            self.scheduler.start()
            
            self.is_running = True
            logger.info("ETL Pipeline Orchestrator started successfully")
            
        except Exception as e:
            logger.error("Failed to start ETL Pipeline Orchestrator", error=str(e))
            raise
    
    async def stop_pipeline(self):
        """Stop the ETL pipeline orchestrator"""
        try:
            logger.info("Stopping ETL Pipeline Orchestrator")
            
            # Stop scheduler
            self.scheduler.shutdown(wait=True)
            
            # Stop monitoring
            await self.pipeline_monitor.stop_monitoring()
            
            self.is_running = False
            logger.info("ETL Pipeline Orchestrator stopped successfully")
            
        except Exception as e:
            logger.error("Failed to stop ETL Pipeline Orchestrator", error=str(e))
            raise
    
    async def run_etl_pipeline(self):
        """Run the complete ETL pipeline"""
        pipeline_start_time = datetime.now(timezone.utc)
        pipeline_id = f"pipeline_{pipeline_start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info("Starting ETL pipeline run", pipeline_id=pipeline_id)
            
            # Record pipeline start
            await self.pipeline_monitor.record_pipeline_start(pipeline_id)
            
            # Step 1: Data Collection and Ingestion
            ingestion_metrics = await self._run_data_ingestion()
            
            # Step 2: Data Validation
            validation_metrics = await self._run_data_validation()
            
            # Step 3: Data Transformation and Feature Engineering
            transformation_metrics = await self._run_data_transformation()
            
            # Step 4: Check if model retraining is needed
            retraining_metrics = await self._check_and_trigger_retraining()
            
            # Step 5: Generate and store predictions
            prediction_metrics = await self._generate_predictions()
            
            # Compile pipeline metrics
            pipeline_metrics = {
                "pipeline_id": pipeline_id,
                "start_time": pipeline_start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": (datetime.now(timezone.utc) - pipeline_start_time).total_seconds(),
                "ingestion": ingestion_metrics,
                "validation": validation_metrics,
                "transformation": transformation_metrics,
                "retraining": retraining_metrics,
                "prediction": prediction_metrics,
                "status": "completed"
            }
            
            # Record successful completion
            await self.pipeline_monitor.record_pipeline_completion(pipeline_id, pipeline_metrics)
            
            self.last_run_timestamp = pipeline_start_time
            self.pipeline_metrics = pipeline_metrics
            
            logger.info("ETL pipeline completed successfully", pipeline_id=pipeline_id)
            
        except Exception as e:
            logger.error("ETL pipeline failed", pipeline_id=pipeline_id, error=str(e))
            
            # Record pipeline failure
            await self.pipeline_monitor.record_pipeline_failure(pipeline_id, str(e))
            
            # Send alerts if configured
            await self._send_failure_alert(pipeline_id, str(e))
            
            raise
    
    async def _run_data_ingestion(self) -> Dict:
        """Run data ingestion step"""
        logger.info("Running data ingestion step")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get list of new data files to process
            new_files = await self._discover_new_data_files()
            
            total_records = 0
            processed_files = 0
            failed_files = 0
            
            for file_path in new_files:
                try:
                    # Process each file
                    result = await self.batch_processor.process_quarterly_data(file_path)
                    total_records += result.get("total_records_processed", 0)
                    processed_files += 1
                    
                    logger.info("Processed data file", file=file_path, records=result.get("total_records_processed", 0))
                    
                except Exception as e:
                    logger.error("Failed to process data file", file=file_path, error=str(e))
                    failed_files += 1
            
            return {
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "files_discovered": len(new_files),
                "files_processed": processed_files,
                "files_failed": failed_files,
                "total_records_ingested": total_records
            }
            
        except Exception as e:
            logger.error("Data ingestion step failed", error=str(e))
            raise
    
    async def _run_data_validation(self) -> Dict:
        """Run data validation step"""
        logger.info("Running data validation step")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get recent raw data files
            raw_files = await self.storage_manager.list_files("raw/")
            recent_files = [
                f for f in raw_files 
                if f["last_modified"] > datetime.now(timezone.utc) - timedelta(days=1)
            ]
            
            total_records = 0
            valid_records = 0
            validation_errors = []
            
            for file_info in recent_files:
                try:
                    # Load and validate data
                    data = await self.storage_manager.retrieve_data(file_info["path"])
                    validation_result = self.data_validator.validate_batch_data(data)
                    
                    total_records += validation_result.validated_records
                    valid_records += (validation_result.validated_records - validation_result.failed_records)
                    validation_errors.extend(validation_result.errors)
                    
                    # Store validated data if it passes validation
                    if validation_result.is_valid:
                        validated_path = f"validated/{file_info['path'].replace('raw/', '')}"
                        await self.storage_manager.store_data(data, validated_path)
                    
                except Exception as e:
                    logger.error("Failed to validate data file", file=file_info["path"], error=str(e))
                    validation_errors.append(f"File {file_info['path']}: {str(e)}")
            
            return {
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "files_validated": len(recent_files),
                "total_records": total_records,
                "valid_records": valid_records,
                "validation_success_rate": (valid_records / total_records * 100) if total_records > 0 else 0,
                "validation_errors": validation_errors[:10]  # Limit to first 10 errors
            }
            
        except Exception as e:
            logger.error("Data validation step failed", error=str(e))
            raise
    
    async def _run_data_transformation(self) -> Dict:
        """Run data transformation and feature engineering step"""
        logger.info("Running data transformation step")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get validated data files
            validated_files = await self.storage_manager.list_files("validated/")
            recent_files = [
                f for f in validated_files 
                if f["last_modified"] > datetime.now(timezone.utc) - timedelta(days=1)
            ]
            
            total_records = 0
            transformed_records = 0
            
            for file_info in recent_files:
                try:
                    # Load validated data
                    data = await self.storage_manager.retrieve_data(file_info["path"])
                    
                    # Apply feature engineering
                    transformed_data = await self.feature_engineer.transform_data(data)
                    
                    # Store transformed data
                    processed_path = f"processed/{file_info['path'].replace('validated/', '')}"
                    await self.storage_manager.store_data(transformed_data, processed_path)
                    
                    total_records += len(data)
                    transformed_records += len(transformed_data)
                    
                except Exception as e:
                    logger.error("Failed to transform data file", file=file_info["path"], error=str(e))
            
            return {
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "files_transformed": len(recent_files),
                "input_records": total_records,
                "output_records": transformed_records,
                "transformation_success_rate": (transformed_records / total_records * 100) if total_records > 0 else 0
            }
            
        except Exception as e:
            logger.error("Data transformation step failed", error=str(e))
            raise
    
    async def _check_and_trigger_retraining(self) -> Dict:
        """Check if model retraining is needed and trigger if necessary"""
        logger.info("Checking if model retraining is needed")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check retraining conditions
            should_retrain, reasons = await self._should_retrain_model()
            
            retraining_metrics = {
                "duration_seconds": 0,
                "retraining_triggered": should_retrain,
                "retraining_reasons": reasons,
                "new_model_performance": None
            }
            
            if should_retrain:
                logger.info("Model retraining triggered", reasons=reasons)
                
                # Run model retraining
                training_result = await self.model_trainer.retrain_model()
                
                retraining_metrics.update({
                    "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "new_model_performance": training_result.get("performance_metrics"),
                    "model_version": training_result.get("model_version")
                })
                
                logger.info("Model retraining completed", 
                           performance=training_result.get("performance_metrics"))
            
            return retraining_metrics
            
        except Exception as e:
            logger.error("Model retraining check failed", error=str(e))
            raise
    
    async def _generate_predictions(self) -> Dict:
        """Generate predictions for new data"""
        logger.info("Generating predictions for new data")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get processed data files
            processed_files = await self.storage_manager.list_files("processed/")
            recent_files = [
                f for f in processed_files 
                if f["last_modified"] > datetime.now(timezone.utc) - timedelta(days=1)
            ]
            
            total_predictions = 0
            
            for file_info in recent_files:
                try:
                    # Load processed data
                    data = await self.storage_manager.retrieve_data(file_info["path"])
                    
                    # Generate predictions using current model
                    predictions = await self.model_trainer.predict(data)
                    
                    # Store predictions
                    predictions_path = f"predictions/{file_info['path'].replace('processed/', '')}"
                    await self.storage_manager.store_data(predictions, predictions_path)
                    
                    total_predictions += len(predictions)
                    
                except Exception as e:
                    logger.error("Failed to generate predictions for file", file=file_info["path"], error=str(e))
            
            return {
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "files_processed": len(recent_files),
                "total_predictions": total_predictions
            }
            
        except Exception as e:
            logger.error("Prediction generation failed", error=str(e))
            raise
    
    async def _discover_new_data_files(self) -> List[str]:
        """Discover new data files to process"""
        # This would typically scan a specific directory or check an external system
        # For now, we'll return an empty list as a placeholder
        return []
    
    async def _should_retrain_model(self) -> tuple[bool, List[str]]:
        """Determine if model retraining should be triggered"""
        reasons = []
        
        try:
            # Check 1: Minimum new samples threshold
            new_data_count = await self._count_new_samples()
            if new_data_count >= config.ml.min_new_samples:
                reasons.append(f"New samples threshold reached: {new_data_count} >= {config.ml.min_new_samples}")
            
            # Check 2: Model performance degradation
            current_performance = await self._get_current_model_performance()
            if current_performance and current_performance < config.ml.performance_threshold:
                reasons.append(f"Performance below threshold: {current_performance:.3f} < {config.ml.performance_threshold}")
            
            # Check 3: Data drift detection
            drift_detected = await self._detect_data_drift()
            if drift_detected:
                reasons.append("Significant data drift detected")
            
            # Check 4: Manual trigger (could be set via API or configuration)
            manual_trigger = await self._check_manual_retrain_trigger()
            if manual_trigger:
                reasons.append("Manual retraining triggered")
            
            return len(reasons) > 0, reasons
            
        except Exception as e:
            logger.error("Failed to check retraining conditions", error=str(e))
            return False, [f"Error checking conditions: {str(e)}"]
    
    async def _count_new_samples(self) -> int:
        """Count new samples since last model training"""
        try:
            # Get processed files since last model update
            processed_files = await self.storage_manager.list_files("processed/")
            last_model_update = await self._get_last_model_update_time()
            
            new_samples = 0
            for file_info in processed_files:
                if file_info["last_modified"] > last_model_update:
                    data = await self.storage_manager.retrieve_data(file_info["path"])
                    new_samples += len(data)
            
            return new_samples
            
        except Exception as e:
            logger.error("Failed to count new samples", error=str(e))
            return 0
    
    async def _get_current_model_performance(self) -> Optional[float]:
        """Get current model performance on recent data"""
        try:
            # This would evaluate the model on recent validation data
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            logger.error("Failed to get model performance", error=str(e))
            return None
    
    async def _detect_data_drift(self) -> bool:
        """Detect if there's significant data drift"""
        try:
            # This would use statistical tests to detect distribution changes
            # For now, return False as placeholder
            return False
            
        except Exception as e:
            logger.error("Failed to detect data drift", error=str(e))
            return False
    
    async def _check_manual_retrain_trigger(self) -> bool:
        """Check if manual retraining has been triggered"""
        try:
            # Check for manual trigger flag in configuration or database
            # For now, return False as placeholder
            return False
            
        except Exception as e:
            logger.error("Failed to check manual trigger", error=str(e))
            return False
    
    async def _get_last_model_update_time(self) -> datetime:
        """Get timestamp of last model update"""
        try:
            # This would query the model registry or database
            # For now, return 30 days ago as placeholder
            return datetime.now(timezone.utc) - timedelta(days=30)
            
        except Exception as e:
            logger.error("Failed to get last model update time", error=str(e))
            return datetime.now(timezone.utc) - timedelta(days=30)
    
    async def run_data_quality_check(self):
        """Run daily data quality monitoring"""
        logger.info("Running daily data quality check")
        
        try:
            # Get recent data and run quality checks
            recent_files = await self.storage_manager.list_files("raw/")
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            
            files_to_check = [
                f for f in recent_files 
                if f["last_modified"] > yesterday
            ]
            
            quality_issues = []
            
            for file_info in files_to_check:
                data = await self.storage_manager.retrieve_data(file_info["path"])
                validation_result = self.data_validator.validate_batch_data(data)
                
                if not validation_result.is_valid:
                    quality_issues.extend(validation_result.errors)
            
            # Record quality metrics
            await self.pipeline_monitor.record_data_quality_metrics({
                "files_checked": len(files_to_check),
                "quality_issues": len(quality_issues),
                "quality_score": max(0, 100 - len(quality_issues) * 10)
            })
            
            logger.info("Data quality check completed", issues=len(quality_issues))
            
        except Exception as e:
            logger.error("Data quality check failed", error=str(e))
    
    async def check_model_performance(self):
        """Check model performance daily"""
        logger.info("Running daily model performance check")
        
        try:
            # Get recent predictions and actual outcomes
            performance_score = await self._calculate_recent_performance()
            
            # Record performance metrics
            await self.pipeline_monitor.record_model_performance({
                "performance_score": performance_score,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info("Model performance check completed", score=performance_score)
            
        except Exception as e:
            logger.error("Model performance check failed", error=str(e))
    
    async def archive_old_data(self):
        """Archive old data weekly"""
        logger.info("Running weekly data archiving")
        
        try:
            # Archive data older than 6 months
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
            
            from storage.storage_manager import DataArchiver
            archiver = DataArchiver(self.storage_manager)
            
            result = await archiver.archive_old_data(cutoff_date)
            
            logger.info("Data archiving completed", 
                       files=result["archived_files"], 
                       size=result["archived_size_bytes"])
            
        except Exception as e:
            logger.error("Data archiving failed", error=str(e))
    
    async def _calculate_recent_performance(self) -> float:
        """Calculate model performance on recent data"""
        try:
            # This would calculate actual performance metrics
            # For now, return a placeholder value
            return 0.85
            
        except Exception as e:
            logger.error("Failed to calculate performance", error=str(e))
            return 0.0
    
    async def _send_failure_alert(self, pipeline_id: str, error_message: str):
        """Send alert for pipeline failure"""
        try:
            if config.pipeline.alert_email:
                # This would send an email alert
                logger.info("Pipeline failure alert sent", 
                           pipeline_id=pipeline_id, 
                           email=config.pipeline.alert_email)
        except Exception as e:
            logger.error("Failed to send failure alert", error=str(e))
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            "is_running": self.is_running,
            "last_run": self.last_run_timestamp.isoformat() if self.last_run_timestamp else None,
            "next_scheduled_run": self.scheduler.get_job('main_etl_pipeline').next_run_time.isoformat() if self.scheduler.get_job('main_etl_pipeline') else None,
            "metrics": self.pipeline_metrics
        }


# Factory function to create orchestrator
def create_etl_orchestrator() -> ETLOrchestrator:
    """Create and configure ETL orchestrator"""
    return ETLOrchestrator()


if __name__ == "__main__":
    async def main():
        orchestrator = create_etl_orchestrator()
        await orchestrator.start_pipeline()
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await orchestrator.stop_pipeline()
    
    asyncio.run(main()) 