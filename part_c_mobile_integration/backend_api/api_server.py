#!/usr/bin/env python3
"""
Part C: Backend API Server for WorkMate Mobile App Integration
RTV Senior Data Scientist Technical Assessment

This module provides:
1. RESTful API endpoints for mobile app integration
2. Model serving and prediction endpoints
3. Data synchronization services
4. Model update and management endpoints
5. Analytics and monitoring APIs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import pickle
import gzip
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic Models for API
class HouseholdData(BaseModel):
    """Household data model for API"""
    household_id: str = Field(..., description="Unique household identifier")
    district: str = Field(..., description="District name")
    cluster: str = Field(..., description="Cluster identifier")
    village: str = Field(..., description="Village name")
    household_size: int = Field(..., ge=1, le=20, description="Number of household members")
    agriculture_land: float = Field(..., ge=0, le=50, description="Agricultural land in acres")
    vsla_profits: float = Field(..., ge=0, le=10000, description="VSLA profits per month")
    business_income: float = Field(..., ge=0, le=5000, description="Business income per month")
    formal_employment: int = Field(..., ge=0, le=1, description="Formal employment status (0/1)")
    time_to_opd: int = Field(..., ge=0, le=300, description="Time to OPD in minutes")
    season1_crops_planted: int = Field(..., ge=0, le=10, description="Number of crops planted in season 1")
    vehicle_owner: int = Field(..., ge=0, le=1, description="Vehicle ownership status (0/1)")
    field_officer_id: str = Field(..., description="Field officer identifier")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('district')
    def validate_district(cls, v):
        valid_districts = ['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo']
        if v not in valid_districts:
            raise ValueError(f'District must be one of: {valid_districts}')
        return v

class PredictionRequest(BaseModel):
    """Prediction request model"""
    household_data: HouseholdData
    model_version: Optional[str] = "latest"
    include_confidence: bool = True
    include_recommendations: bool = True

class PredictionResponse(BaseModel):
    """Prediction response model"""
    household_id: str
    vulnerability_class: str
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    recommendations: List[str]
    model_version: str
    prediction_time: str
    processing_time_ms: float

class BatchSyncRequest(BaseModel):
    """Batch synchronization request"""
    households: List[HouseholdData]
    field_officer_id: str
    sync_timestamp: str
    device_id: str

class BatchSyncResponse(BaseModel):
    """Batch synchronization response"""
    sync_id: str
    total_records: int
    successful_records: int
    failed_records: int
    sync_timestamp: str
    errors: List[Dict[str, Any]]

class ModelUpdateInfo(BaseModel):
    """Model update information"""
    current_version: str
    latest_version: str
    update_available: bool
    update_size_mb: float
    release_notes: List[str]
    download_url: Optional[str]

class AnalyticsData(BaseModel):
    """Analytics data model"""
    metric_name: str
    metric_value: Any
    timestamp: str
    field_officer_id: Optional[str]
    household_id: Optional[str]
    additional_data: Optional[Dict[str, Any]]

# Global variables for model and data storage
prediction_model = None
model_metadata = {}
analytics_store = []
sync_records = {}

class PredictionService:
    """Handles ML model predictions"""
    
    def __init__(self):
        self.model = None
        self.model_version = "2.0.0-api"
        self.load_model()
    
    def load_model(self):
        """Load the ML model for predictions"""
        try:
            # Try to load model from Part A or model packaging
            model_paths = [
                "../part_a_predictive_modeling/best_vulnerability_model.pkl",
                "mobile_models/household_vulnerability_mobile_v2.pkl.gz"
            ]
            
            for model_path in model_paths:
                if Path(model_path).exists():
                    logger.info(f"Loading model from: {model_path}")
                    
                    if model_path.endswith('.gz'):
                        with gzip.open(model_path, 'rb') as f:
                            self.model = pickle.load(f)
                    else:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f)
                    
                    logger.info("✅ Model loaded successfully")
                    return
            
            # If no model found, create demo model
            logger.warning("No trained model found, creating demo prediction service")
            self.model = self._create_demo_model()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_demo_model()
    
    def _create_demo_model(self):
        """Create demo model for API testing"""
        return {
            'type': 'demo_api_model',
            'version': self.model_version,
            'vulnerability_classes': ["On Track", "At Risk", "Struggling", "Severely Struggling"]
        }
    
    async def predict(self, household_data: HouseholdData) -> PredictionResponse:
        """Generate prediction for household data"""
        start_time = datetime.now()
        
        try:
            # Preprocess data
            features = self._preprocess_household_data(household_data)
            
            # Generate prediction
            if isinstance(self.model, dict) and self.model.get('type') == 'demo_api_model':
                # Demo prediction logic
                vulnerability_class, risk_level, confidence, probabilities = self._demo_predict(features)
            else:
                # Real model prediction
                vulnerability_class, risk_level, confidence, probabilities = self._model_predict(features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(vulnerability_class, features)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = PredictionResponse(
                household_id=household_data.household_id,
                vulnerability_class=vulnerability_class,
                risk_level=risk_level,
                confidence=confidence,
                probabilities=probabilities,
                recommendations=recommendations,
                model_version=self.model_version,
                prediction_time=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def _preprocess_household_data(self, household_data: HouseholdData) -> Dict[str, float]:
        """Preprocess household data for prediction"""
        # District encoding
        district_mapping = {'Mitooma': 0, 'Kanungu': 1, 'Rubirizi': 2, 'Ntungamo': 3}
        
        features = {
            'household_size': float(household_data.household_size),
            'agriculture_land': float(household_data.agriculture_land),
            'vsla_profits': float(household_data.vsla_profits),
            'business_income': float(household_data.business_income),
            'formal_employment': float(household_data.formal_employment),
            'time_to_opd': float(household_data.time_to_opd),
            'season1_crops_planted': float(household_data.season1_crops_planted),
            'vehicle_owner': float(household_data.vehicle_owner),
            'district_encoded': float(district_mapping.get(household_data.district, 0))
        }
        
        return features
    
    def _demo_predict(self, features: Dict[str, float]) -> tuple:
        """Demo prediction logic"""
        # Calculate vulnerability score
        score = 0
        
        if features['household_size'] > 8:
            score += 1
        if features['agriculture_land'] < 1.0:
            score += 1
        if features['vsla_profits'] < 200:
            score += 1
        if features['business_income'] < 150:
            score += 1
        if features['formal_employment'] == 0:
            score += 0.5
        if features['time_to_opd'] > 120:
            score += 1
        if features['season1_crops_planted'] < 2:
            score += 0.5
        
        # Map to vulnerability class
        vulnerability_classes = ["On Track", "At Risk", "Struggling", "Severely Struggling"]
        
        if score >= 3:
            class_idx = 3
            risk_level = "Critical"
            confidence = 0.92
        elif score >= 2:
            class_idx = 2
            risk_level = "High"
            confidence = 0.88
        elif score >= 1:
            class_idx = 1
            risk_level = "Medium"
            confidence = 0.85
        else:
            class_idx = 0
            risk_level = "Low"
            confidence = 0.90
        
        # Generate probabilities
        probabilities = {cls: 0.1 for cls in vulnerability_classes}
        probabilities[vulnerability_classes[class_idx]] = confidence
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return vulnerability_classes[class_idx], risk_level, confidence, probabilities
    
    def _model_predict(self, features: Dict[str, float]) -> tuple:
        """Real model prediction"""
        try:
            # Convert features to the format expected by the model
            feature_array = [
                features['household_size'],
                features['agriculture_land'],
                features['vsla_profits'],
                features['business_income'],
                features['formal_employment'],
                features['time_to_opd'],
                features['season1_crops_planted'],
                features['vehicle_owner'],
                features['district_encoded']
            ]
            
            # Get prediction from actual model
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([feature_array])[0]
                prediction_class = probabilities.argmax()
                confidence = float(probabilities.max())
            else:
                # Fallback to demo prediction
                return self._demo_predict(features)
            
            vulnerability_classes = ["On Track", "At Risk", "Struggling", "Severely Struggling"]
            vulnerability_class = vulnerability_classes[prediction_class]
            
            # Determine risk level
            risk_levels = ["Low", "Medium", "High", "Critical"]
            risk_level = risk_levels[prediction_class]
            
            # Format probabilities
            prob_dict = {
                vulnerability_classes[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return vulnerability_class, risk_level, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return self._demo_predict(features)
    
    def _generate_recommendations(self, vulnerability_class: str, features: Dict[str, float]) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []
        
        if vulnerability_class == "Severely Struggling":
            recommendations.extend([
                "Immediate cash transfer assistance required",
                "Emergency food support enrollment",
                "Healthcare access facilitation",
                "Connect with emergency services"
            ])
        elif vulnerability_class == "Struggling":
            recommendations.extend([
                "Enroll in targeted livelihood programs",
                "Business training and support",
                "Agricultural extension services",
                "VSLA group participation"
            ])
        elif vulnerability_class == "At Risk":
            recommendations.extend([
                "Preventive program enrollment",
                "Savings group formation",
                "Skills training opportunities",
                "Regular monitoring visits"
            ])
        else:
            recommendations.extend([
                "Community program participation",
                "Peer support networks",
                "Economic opportunity sharing"
            ])
        
        # Add specific recommendations based on features
        if features['agriculture_land'] < 1.0:
            recommendations.append("Land productivity improvement training")
        
        if features['season1_crops_planted'] < 2:
            recommendations.append("Crop diversification support")
        
        if features['time_to_opd'] > 120:
            recommendations.append("Healthcare access improvement needed")
        
        return recommendations[:5]  # Limit to top 5

# Initialize services
prediction_service = PredictionService()

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting WorkMate API Server")
    logger.info("✅ Prediction service initialized")
    yield
    # Shutdown
    logger.info("📪 Shutting down WorkMate API Server")

app = FastAPI(
    title="WorkMate API",
    description="Backend API for RTV WorkMate Mobile App - Household Vulnerability Assessment",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    # In production, implement proper JWT token validation
    if not credentials.credentials:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"user_id": "demo_user", "role": "field_officer"}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "WorkMate API",
        "version": "2.0.0",
        "description": "Backend API for RTV Household Vulnerability Assessment",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "prediction": "/api/v1/predict",
            "sync": "/api/v1/sync/batch",
            "model_update": "/api/v1/models/update-info",
            "analytics": "/api/v1/analytics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": prediction_service.model is not None,
        "model_version": prediction_service.model_version,
        "uptime": "operational"
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_vulnerability(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate vulnerability prediction for a household"""
    try:
        logger.info(f"Prediction request for household: {request.household_data.household_id}")
        
        # Generate prediction
        prediction = await prediction_service.predict(request.household_data)
        
        # Log analytics
        analytics_store.append({
            "metric_name": "prediction_generated",
            "timestamp": datetime.now().isoformat(),
            "household_id": request.household_data.household_id,
            "field_officer_id": request.household_data.field_officer_id,
            "vulnerability_class": prediction.vulnerability_class,
            "confidence": prediction.confidence
        })
        
        logger.info(f"✅ Prediction completed: {prediction.vulnerability_class} ({prediction.confidence:.1%})")
        return prediction
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sync/batch", response_model=BatchSyncResponse)
async def sync_batch_data(
    request: BatchSyncRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Synchronize batch data from mobile devices"""
    try:
        sync_id = str(uuid.uuid4())
        logger.info(f"Batch sync request: {sync_id} - {len(request.households)} households")
        
        successful_records = 0
        failed_records = 0
        errors = []
        
        # Process each household
        for household in request.households:
            try:
                # Validate and store household data
                # In production, save to database
                sync_records[household.household_id] = {
                    "household_data": household.dict(),
                    "sync_id": sync_id,
                    "synced_at": datetime.now().isoformat(),
                    "field_officer_id": request.field_officer_id
                }
                
                successful_records += 1
                
                # Generate prediction asynchronously
                background_tasks.add_task(
                    generate_background_prediction,
                    household
                )
                
            except Exception as e:
                failed_records += 1
                errors.append({
                    "household_id": household.household_id,
                    "error": str(e)
                })
                logger.error(f"Error processing household {household.household_id}: {e}")
        
        # Log analytics
        analytics_store.append({
            "metric_name": "batch_sync_completed",
            "timestamp": datetime.now().isoformat(),
            "sync_id": sync_id,
            "field_officer_id": request.field_officer_id,
            "total_records": len(request.households),
            "successful_records": successful_records,
            "failed_records": failed_records
        })
        
        response = BatchSyncResponse(
            sync_id=sync_id,
            total_records=len(request.households),
            successful_records=successful_records,
            failed_records=failed_records,
            sync_timestamp=datetime.now().isoformat(),
            errors=errors
        )
        
        logger.info(f"✅ Batch sync completed: {successful_records}/{len(request.households)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/update-info", response_model=ModelUpdateInfo)
async def get_model_update_info(
    current_version: str = "2.0.0",
    current_user: dict = Depends(get_current_user)
):
    """Get model update information"""
    try:
        # In production, check against model registry
        latest_version = "2.1.0"
        update_available = current_version != latest_version
        
        update_info = ModelUpdateInfo(
            current_version=current_version,
            latest_version=latest_version,
            update_available=update_available,
            update_size_mb=2.3,
            release_notes=[
                "Improved accuracy for rural household classification",
                "Enhanced confidence calibration",
                "Reduced model size by 15%",
                "Better handling of missing data"
            ],
            download_url="https://api.rtv.org/models/v2.1.0/download" if update_available else None
        )
        
        return update_info
        
    except Exception as e:
        logger.error(f"Error getting model update info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analytics")
async def log_analytics(
    data: AnalyticsData,
    current_user: dict = Depends(get_current_user)
):
    """Log analytics data from mobile app"""
    try:
        # Store analytics data
        analytics_store.append(data.dict())
        
        logger.info(f"Analytics logged: {data.metric_name}")
        return {"status": "success", "message": "Analytics data logged"}
        
    except Exception as e:
        logger.error(f"Error logging analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(
    field_officer_id: Optional[str] = None,
    days: int = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics summary"""
    try:
        # Filter analytics data
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_data = [
            record for record in analytics_store
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if field_officer_id:
            filtered_data = [
                record for record in filtered_data
                if record.get('field_officer_id') == field_officer_id
            ]
        
        # Calculate summary statistics
        total_predictions = len([r for r in filtered_data if r['metric_name'] == 'prediction_generated'])
        total_syncs = len([r for r in filtered_data if r['metric_name'] == 'batch_sync_completed'])
        
        vulnerability_distribution = {}
        for record in filtered_data:
            if record['metric_name'] == 'prediction_generated':
                vc = record.get('vulnerability_class', 'Unknown')
                vulnerability_distribution[vc] = vulnerability_distribution.get(vc, 0) + 1
        
        summary = {
            "period_days": days,
            "total_predictions": total_predictions,
            "total_syncs": total_syncs,
            "vulnerability_distribution": vulnerability_distribution,
            "field_officer_id": field_officer_id,
            "generated_at": datetime.now().isoformat()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sync/status/{sync_id}")
async def get_sync_status(
    sync_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get synchronization status"""
    try:
        # Count records for this sync_id
        records = [r for r in sync_records.values() if r.get('sync_id') == sync_id]
        
        if not records:
            raise HTTPException(status_code=404, detail="Sync ID not found")
        
        status = {
            "sync_id": sync_id,
            "total_records": len(records),
            "status": "completed",
            "synced_at": records[0]['synced_at'] if records else None,
            "field_officer_id": records[0]['field_officer_id'] if records else None
        }
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def generate_background_prediction(household_data: HouseholdData):
    """Generate prediction in background after sync"""
    try:
        prediction = await prediction_service.predict(household_data)
        
        # Store prediction result
        sync_records[household_data.household_id]['prediction'] = prediction.dict()
        
        logger.info(f"Background prediction completed for {household_data.household_id}")
        
    except Exception as e:
        logger.error(f"Error in background prediction: {e}")

# Demo and testing endpoints
@app.post("/api/v1/demo/generate-test-data")
async def generate_test_data():
    """Generate test data for API demonstration"""
    try:
        test_households = []
        
        # Generate sample households
        import random
        districts = ['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo']
        
        for i in range(5):
            household = HouseholdData(
                household_id=f"TEST{i+1:03d}",
                district=random.choice(districts),
                cluster=f"CL{i+1:03d}",
                village=f"Village_{i+1}",
                household_size=random.randint(3, 12),
                agriculture_land=round(random.uniform(0.5, 5.0), 2),
                vsla_profits=random.randint(100, 800),
                business_income=random.randint(50, 500),
                formal_employment=random.choice([0, 1]),
                time_to_opd=random.randint(30, 180),
                season1_crops_planted=random.randint(1, 6),
                vehicle_owner=random.choice([0, 1]),
                field_officer_id="DEMO_FO"
            )
            test_households.append(household)
        
        # Generate predictions for test data
        predictions = []
        for household in test_households:
            prediction = await prediction_service.predict(household)
            predictions.append(prediction)
        
        return {
            "test_households": [h.dict() for h in test_households],
            "predictions": [p.dict() for p in predictions],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating test data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main application runner
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting WorkMate API Server")
    logger.info("📖 API Documentation: http://localhost:8000/docs")
    logger.info("💻 API Base URL: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    ) 