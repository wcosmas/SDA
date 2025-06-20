"""
Feature Engineering Module for RTV Household Vulnerability Assessment
Part B: Data Engineering Pipeline

This module handles feature engineering and data transformation for the vulnerability
prediction model based on the ML work from Part A.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """Feature engineering and data transformation for vulnerability prediction"""
    
    def __init__(self):
        """Initialize the feature engineer with predefined feature sets"""
        
        # Feature categories based on Part A analysis
        self.numeric_features = [
            'HouseholdSize', 'TimeToOPD', 'TimeToWater', 'AgricultureLand',
            'Season1CropsPlanted', 'Season2CropsPlanted', 'PerennialCropsGrown',
            'Season1VegetableIncome', 'Season2VegatableIncome', 'VegetableIncome',
            'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 'CasualLabour',
            'RemittancesAndGifts', 'RentIncome', 'SeasonalCropIncome',
            'PerenialCropIncome', 'LivestockIncome', 'AgricValue',
            'HouseholdIcome', 'Assets.1'
        ]
        
        self.categorical_features = [
            'District', 'hhh_sex', 'hhh_read_write', 'Material_walls'
        ]
        
        self.binary_features = [
            'radios_owned', 'phones_owned', 'work_casual', 'work_salaried',
            'latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
            'standard_hangline', 'kitchen_house', 'bathroom_constructed',
            'swept_compound', 'dish_rack_present', 'perennial_cropping',
            'household_fertilizer', 'non_bio_waste_mgt_present',
            'apply_liquid_manure', 'water_control_practise', 'soil_management',
            'postharvest_food_storage', 'save_mode_7'
        ]
        
        self.preprocessor = None
        self.feature_names = None
        
    async def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into model-ready features
        
        Args:
            data: Raw household survey data
            
        Returns:
            Transformed dataframe ready for ML prediction
        """
        try:
            logger.info("Starting feature engineering transformation", 
                       input_shape=data.shape)
            
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Step 1: Feature Engineering
            df = self._engineer_features(df)
            
            # Step 2: Feature Selection
            df = self._select_available_features(df)
            
            # Step 3: Create preprocessing pipeline if not exists
            if self.preprocessor is None:
                self.preprocessor = self._create_preprocessing_pipeline(df)
            
            # Step 4: Apply preprocessing
            transformed_data = self._apply_preprocessing(df)
            
            logger.info("Feature engineering completed successfully",
                       output_shape=transformed_data.shape)
            
            return transformed_data
            
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e))
            raise
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features based on Part A analysis"""
        
        logger.info("Engineering derived features")
        
        # Income per capita
        if 'HouseholdSize' in df.columns and 'HouseholdIcome' in df.columns:
            df['income_per_capita'] = df['HouseholdIcome'] / df['HouseholdSize'].replace(0, 1)
            self.numeric_features.append('income_per_capita')
            logger.info("Created income_per_capita feature")
        
        # Agricultural productivity
        if 'AgricultureLand' in df.columns and 'AgricValue' in df.columns:
            df['agric_productivity'] = df['AgricValue'] / (df['AgricultureLand'].replace(0, 0.1))
            self.numeric_features.append('agric_productivity')
            logger.info("Created agric_productivity feature")
        
        # Household size categories
        if 'HouseholdSize' in df.columns:
            df['household_size_category'] = pd.cut(
                df['HouseholdSize'], 
                bins=[0, 3, 5, 7, float('inf')], 
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
            self.categorical_features.append('household_size_category')
            logger.info("Created household_size_category feature")
        
        # Asset ownership score
        asset_cols = ['radios_owned', 'phones_owned']
        available_asset_cols = [col for col in asset_cols if col in df.columns]
        if available_asset_cols:
            df['asset_ownership_score'] = df[available_asset_cols].sum(axis=1)
            self.numeric_features.append('asset_ownership_score')
            logger.info("Created asset_ownership_score feature")
        
        # Infrastructure access score
        infra_cols = [
            'latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
            'bathroom_constructed', 'kitchen_house'
        ]
        available_infra_cols = [col for col in infra_cols if col in df.columns]
        if available_infra_cols:
            df['infrastructure_score'] = df[available_infra_cols].sum(axis=1)
            self.numeric_features.append('infrastructure_score')
            logger.info("Created infrastructure_score feature")
        
        # Total employment income
        employment_cols = [
            'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 
            'CasualLabour', 'RemittancesAndGifts'
        ]
        available_employment_cols = [col for col in employment_cols if col in df.columns]
        if available_employment_cols:
            df['total_employment_income'] = df[available_employment_cols].sum(axis=1)
            self.numeric_features.append('total_employment_income')
            logger.info("Created total_employment_income feature")
        
        # Total agricultural income
        agric_income_cols = [
            'SeasonalCropIncome', 'PerenialCropIncome', 'LivestockIncome', 
            'VegetableIncome'
        ]
        available_agric_cols = [col for col in agric_income_cols if col in df.columns]
        if available_agric_cols:
            df['total_agricultural_income'] = df[available_agric_cols].sum(axis=1)
            self.numeric_features.append('total_agricultural_income')
            logger.info("Created total_agricultural_income feature")
        
        # Livelihood diversity score
        livelihood_cols = available_employment_cols + available_agric_cols
        if livelihood_cols:
            df['livelihood_diversity'] = (df[livelihood_cols] > 0).sum(axis=1)
            self.numeric_features.append('livelihood_diversity')
            logger.info("Created livelihood_diversity feature")
        
        return df
    
    def _select_available_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter features to only include those available in the dataset"""
        
        # Filter features that exist in the dataset
        available_numeric = [f for f in self.numeric_features if f in df.columns]
        available_categorical = [f for f in self.categorical_features if f in df.columns]
        available_binary = [f for f in self.binary_features if f in df.columns]
        
        # Update feature lists
        self.numeric_features = available_numeric
        self.categorical_features = available_categorical
        self.binary_features = available_binary
        
        # Select only available features
        all_features = self.numeric_features + self.categorical_features + self.binary_features
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info("Feature availability check completed",
                   numeric_features=len(self.numeric_features),
                   categorical_features=len(self.categorical_features),
                   binary_features=len(self.binary_features),
                   total_features=len(available_features))
        
        return df[available_features]
    
    def _create_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline for the features"""
        
        logger.info("Creating preprocessing pipeline")
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Binary preprocessing (treat as numeric but don't scale)
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bin', binary_transformer, self.binary_features)
            ],
            remainder='drop'
        )
        
        logger.info("Preprocessing pipeline created successfully")
        
        return preprocessor
    
    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline to the data"""
        
        logger.info("Applying preprocessing pipeline")
        
        # Fit and transform the data
        transformed_array = self.preprocessor.fit_transform(df)
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names()
        
        # Create dataframe from transformed array
        transformed_df = pd.DataFrame(
            transformed_array, 
            columns=feature_names,
            index=df.index
        )
        
        logger.info("Preprocessing applied successfully",
                   original_features=len(df.columns),
                   transformed_features=len(feature_names))
        
        return transformed_df
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        
        if self.feature_names is not None:
            return self.feature_names
        
        feature_names = []
        
        # Numeric features
        feature_names.extend(self.numeric_features)
        
        # Categorical features (after one-hot encoding)
        if self.categorical_features:
            try:
                cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)
            except:
                # Fallback if get_feature_names_out is not available
                for cat_feat in self.categorical_features:
                    feature_names.append(f"{cat_feat}_encoded")
        
        # Binary features
        feature_names.extend(self.binary_features)
        
        self.feature_names = feature_names
        return feature_names
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features used"""
        
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "binary_features": self.binary_features,
            "total_features": len(self.numeric_features) + len(self.categorical_features) + len(self.binary_features),
            "feature_names": self.feature_names
        }
    
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for required features
        
        Args:
            data: Input dataframe to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for minimum required features
        required_features = ['HouseholdSize', 'HouseholdIcome', 'District']
        missing_required = [f for f in required_features if f not in data.columns]
        
        if missing_required:
            issues.append(f"Missing required features: {missing_required}")
        
        # Check data quality
        if data.empty:
            issues.append("Input data is empty")
        
        if data.isnull().all().any():
            null_cols = data.columns[data.isnull().all()].tolist()
            issues.append(f"Columns with all null values: {null_cols}")
        
        # Check for reasonable data ranges
        if 'HouseholdSize' in data.columns:
            invalid_household_sizes = (data['HouseholdSize'] < 0) | (data['HouseholdSize'] > 50)
            if invalid_household_sizes.any():
                issues.append("Invalid household sizes detected (< 0 or > 50)")
        
        return len(issues) == 0, issues 