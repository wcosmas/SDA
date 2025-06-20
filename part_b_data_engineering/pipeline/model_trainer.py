"""
Model Training Module for RTV Household Vulnerability Assessment
Part B: Data Engineering Pipeline

This module handles model training, retraining, and inference for the vulnerability
prediction system based on the ML work from Part A.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import joblib
import os
from datetime import datetime, timezone
from pathlib import Path
import structlog
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """Model training and inference for vulnerability prediction"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model trainer
        
        Args:
            model_path: Path to existing model file. If None, will look for default model.
        """
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.model_metadata = {}
        self.feature_names = None
        
        # Load existing model if available
        self._load_existing_model()
        
        # Model configurations based on Part A results
        self.model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            }
        }
    
    def _get_default_model_path(self) -> str:
        """Get the default model path"""
        # First try to find the model from Part A
        part_a_model = Path("../part_a_predictive_modeling/best_vulnerability_model_final.pkl")
        if part_a_model.exists():
            return str(part_a_model)
        
        # Otherwise use local model path
        return "models/best_vulnerability_model.pkl"
    
    def _load_existing_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self._load_model_metadata()
                logger.info("Existing model loaded successfully", model_path=self.model_path)
            else:
                logger.info("No existing model found", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load existing model", error=str(e))
    
    def _load_model_metadata(self):
        """Load model metadata"""
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        try:
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded", metadata=self.model_metadata)
        except Exception as e:
            logger.warning("Failed to load model metadata", error=str(e))
    
    async def retrain_model(self, training_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Retrain the model with new data
        
        Args:
            training_data: New training data. If None, will load latest processed data.
            
        Returns:
            Dictionary containing training results and performance metrics
        """
        try:
            logger.info("Starting model retraining process")
            
            # Step 1: Load training data
            if training_data is None:
                training_data = await self._load_latest_training_data()
            
            if training_data is None or training_data.empty:
                raise ValueError("No training data available")
            
            # Step 2: Prepare features and target
            X, y = self._prepare_training_data(training_data)
            
            # Step 3: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Step 4: Train models and select best
            best_model, training_results = await self._train_and_select_best_model(
                X_train, X_test, y_train, y_test
            )
            
            # Step 5: Evaluate final model
            final_metrics = self._evaluate_model(best_model, X_test, y_test)
            
            # Step 6: Save model and metadata
            model_version = await self._save_model(best_model, final_metrics)
            
            # Prepare return results
            results = {
                "model_version": model_version,
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "performance_metrics": final_metrics,
                "training_results": training_results
            }
            
            logger.info("Model retraining completed successfully", results=results)
            
            return results
            
        except Exception as e:
            logger.error("Model retraining failed", error=str(e))
            raise
    
    async def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new data
        
        Args:
            data: Input data for prediction
            
        Returns:
            DataFrame with predictions and probabilities
        """
        try:
            if self.model is None:
                raise ValueError("No model available for prediction")
            
            logger.info("Generating predictions", input_shape=data.shape)
            
            # Generate predictions
            predictions = self.model.predict(data)
            prediction_probabilities = self.model.predict_proba(data)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'household_id': data.index,
                'prediction': predictions,
                'vulnerability_probability': prediction_probabilities[:, 1],
                'confidence_score': np.max(prediction_probabilities, axis=1),
                'prediction_timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Add interpretable labels
            results['vulnerability_status'] = results['prediction'].map({
                0: 'Non-vulnerable',
                1: 'Vulnerable'
            })
            
            # Add risk categories based on probability
            results['risk_category'] = pd.cut(
                results['vulnerability_probability'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
            )
            
            logger.info("Predictions generated successfully", 
                       predictions_count=len(results),
                       vulnerable_count=sum(predictions))
            
            return results
            
        except Exception as e:
            logger.error("Prediction generation failed", error=str(e))
            raise
    
    async def _load_latest_training_data(self) -> Optional[pd.DataFrame]:
        """Load the latest processed training data"""
        try:
            # Look for processed data files
            data_paths = [
                "../part_a_predictive_modeling/data_with_proper_progress_status.csv",
                "data/processed/latest_training_data.csv",
                "data/training/household_data.csv"
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    data = pd.read_csv(path)
                    logger.info("Training data loaded", path=path, shape=data.shape)
                    return data
            
            logger.warning("No training data found in expected locations")
            return None
            
        except Exception as e:
            logger.error("Failed to load training data", error=str(e))
            return None
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from training data"""
        
        # Create target variable if it doesn't exist
        if 'is_vulnerable' not in data.columns:
            if 'ProgressStatus' in data.columns:
                data['is_vulnerable'] = data['ProgressStatus'].isin(['Struggling', 'Severely Struggling']).astype(int)
            else:
                raise ValueError("No target variable found in training data")
        
        # Get target
        y = data['is_vulnerable']
        
        # Remove target and ID columns from features
        feature_cols = [col for col in data.columns 
                       if col not in ['is_vulnerable', 'ProgressStatus', 'household_id', 'ID']]
        X = data[feature_cols]
        
        logger.info("Training data prepared", 
                   features=len(feature_cols), 
                   samples=len(X),
                   vulnerable_ratio=y.mean())
        
        return X, y
    
    async def _train_and_select_best_model(self, X_train, X_test, y_train, y_test) -> Tuple[Any, Dict]:
        """Train multiple models and select the best one"""
        
        logger.info("Training and evaluating multiple models")
        
        from pipeline.feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        results = {}
        best_model = None
        best_score = 0
        
        # Apply feature engineering to training data
        X_train_engineered = await feature_engineer.transform_data(X_train)
        X_test_engineered = await feature_engineer.transform_data(X_test)
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Create pipeline with feature engineering
                pipeline = Pipeline([
                    ('classifier', config['model'])
                ])
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train_engineered, y_train, 
                    cv=5, scoring='f1'
                )
                
                # Fit model
                pipeline.fit(X_train_engineered, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test_engineered)
                y_pred_proba = pipeline.predict_proba(X_test_engineered)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                results[model_name] = {
                    'model': pipeline,
                    'metrics': metrics
                }
                
                # Track best model based on F1 score
                if metrics['f1_score'] > best_score:
                    best_score = metrics['f1_score']
                    best_model = pipeline
                
                logger.info(f"{model_name} training completed", metrics=metrics)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}", error=str(e))
        
        if best_model is None:
            raise ValueError("No models trained successfully")
        
        return best_model, results
    
    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    async def _save_model(self, model, metrics: Dict[str, Any]) -> str:
        """Save model and metadata"""
        
        # Create model directory if it doesn't exist
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"vulnerability_model_{timestamp}"
        
        # Update model path with version
        versioned_model_path = str(model_dir / f"{model_version}.pkl")
        
        # Save model
        joblib.dump(model, versioned_model_path)
        
        # Save metadata
        metadata = {
            'model_version': model_version,
            'created_timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_metrics': metrics,
            'model_type': str(type(model.named_steps['classifier']).__name__),
        }
        
        metadata_path = versioned_model_path.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update current model reference
        self.model = model
        self.model_metadata = metadata
        self.model_path = versioned_model_path
        
        # Create symlink to latest model
        latest_model_path = model_dir / "latest_model.pkl"
        if latest_model_path.exists():
            latest_model_path.unlink()
        latest_model_path.symlink_to(Path(versioned_model_path).name)
        
        logger.info("Model saved successfully", 
                   model_version=model_version,
                   model_path=versioned_model_path)
        
        return model_version
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        if self.model is None:
            return {"status": "no_model_loaded"}
        
        return {
            "status": "model_loaded",
            "model_path": self.model_path,
            "metadata": self.model_metadata,
            "model_type": str(type(self.model.named_steps.get('classifier', self.model)).__name__)
        }
    
    async def validate_model_performance(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate current model performance on new data"""
        
        if self.model is None:
            raise ValueError("No model available for validation")
        
        try:
            # Prepare validation data
            X, y = self._prepare_training_data(validation_data)
            
            # Apply feature engineering
            from pipeline.feature_engineer import FeatureEngineer
            feature_engineer = FeatureEngineer()
            X_engineered = await feature_engineer.transform_data(X)
            
            # Evaluate model
            metrics = self._evaluate_model(self.model, X_engineered, y)
            
            logger.info("Model validation completed", metrics=metrics)
            
            return {
                "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_samples": len(X),
                "performance_metrics": metrics
            }
            
        except Exception as e:
            logger.error("Model validation failed", error=str(e))
            raise 