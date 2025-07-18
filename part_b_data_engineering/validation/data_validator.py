"""
Part B: Data Validation Component
RTV Senior Data Scientist Technical Assessment

This module provides comprehensive data validation for household survey data:
- Schema validation for 75-variable structure
- Data quality checks optimized for high-quality data (97.4% completeness)
- Business rule validation based on assessment thresholds
- Statistical anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import great_expectations as ge
from great_expectations.core import ExpectationSuite
import structlog
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from config.pipeline_config import config

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    validated_records: int
    failed_records: int


@dataclass  
class DriftAnalysisResult:
    """Container for drift analysis results"""
    drift_detected: bool
    overall_drift_score: float
    feature_drift_scores: Dict[str, float]
    significant_features: List[str]
    drift_summary: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation for household survey data with 75-variable structure"""
    
    def __init__(self):
        self.expectations_suite = self._create_expectations_suite()
        self.business_rules = self._load_business_rules()
        self.data_dictionary = self._load_data_dictionary()
        self.reference_data = None  # Will be loaded when needed for drift detection
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data = reference_data
        logger.info("Reference data set for drift detection", samples=len(reference_data))
    
    def analyze_data_drift(self, new_data: pd.DataFrame) -> DriftAnalysisResult:
        """Comprehensive drift analysis using statistical methods"""
        try:
            if self.reference_data is None:
                logger.warning("No reference data available for drift analysis")
                return DriftAnalysisResult(
                    drift_detected=False,
                    overall_drift_score=0.0,
                    feature_drift_scores={},
                    significant_features=[],
                    drift_summary={'error': 'No reference data available'}
                )
            
            logger.info("Starting comprehensive drift analysis", 
                       new_samples=len(new_data),
                       reference_samples=len(self.reference_data))
            
            # Key features for drift analysis
            numeric_features = [
                'HouseholdSize', 'AgricultureLand', 'HHIncome+Consumption+Residues/Day',
                'VSLA_Profits', 'BusinessIncome', 'TimeToOPD', 'TimeToWater'
            ]
            categorical_features = ['District']
            
            feature_drift_scores = {}
            significant_features = []
            drift_details = {}
            
            # 1. Analyze numeric features
            for feature in numeric_features:
                if feature in self.reference_data.columns and feature in new_data.columns:
                    drift_result = self._analyze_numeric_feature_drift(
                        self.reference_data[feature].dropna(),
                        new_data[feature].dropna(),
                        feature
                    )
                    
                    feature_drift_scores[feature] = drift_result['overall_score']
                    drift_details[feature] = drift_result
                    
                    if drift_result['drift_detected']:
                        significant_features.append(feature)
            
            # 2. Analyze categorical features
            for feature in categorical_features:
                if feature in self.reference_data.columns and feature in new_data.columns:
                    drift_result = self._analyze_categorical_feature_drift(
                        self.reference_data[feature],
                        new_data[feature],
                        feature
                    )
                    
                    feature_drift_scores[feature] = drift_result['overall_score']
                    drift_details[feature] = drift_result
                    
                    if drift_result['drift_detected']:
                        significant_features.append(feature)
            
            # 3. Calculate overall drift
            if feature_drift_scores:
                overall_drift_score = np.mean(list(feature_drift_scores.values()))
                max_drift_score = max(feature_drift_scores.values())
                
                # Drift detected if average > threshold or any feature significantly drifted
                drift_detected = (
                    overall_drift_score > config.ml.data_drift_threshold or
                    max_drift_score > 0.25 or
                    len(significant_features) > 0
                )
            else:
                overall_drift_score = 0.0
                drift_detected = False
            
            # 4. Generate drift summary
            drift_summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'features_analyzed': len(feature_drift_scores),
                'significant_drift_features': len(significant_features),
                'max_feature_drift': max(feature_drift_scores.values()) if feature_drift_scores else 0,
                'drift_threshold': config.ml.data_drift_threshold,
                'feature_details': drift_details
            }
            
            logger.info("Drift analysis completed",
                       overall_drift=overall_drift_score,
                       drift_detected=drift_detected,
                       significant_features=len(significant_features))
            
            return DriftAnalysisResult(
                drift_detected=drift_detected,
                overall_drift_score=overall_drift_score,
                feature_drift_scores=feature_drift_scores,
                significant_features=significant_features,
                drift_summary=drift_summary
            )
            
        except Exception as e:
            logger.error("Drift analysis failed", error=str(e))
            return DriftAnalysisResult(
                drift_detected=False,
                overall_drift_score=0.0,
                feature_drift_scores={},
                significant_features=[],
                drift_summary={'error': str(e)}
            )
    
    def _analyze_numeric_feature_drift(self, reference: pd.Series, new: pd.Series, feature_name: str) -> Dict:
        """Analyze drift in numeric feature using multiple statistical tests"""
        try:
            if len(reference) == 0 or len(new) == 0:
                return {'drift_detected': False, 'overall_score': 0.0, 'error': 'Insufficient data'}
            
            # 1. Population Stability Index (PSI)
            psi_score = self._calculate_psi(reference, new)
            
            # 2. Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = self._ks_test(reference, new)
            
            # 3. Mean shift detection (standardized)
            mean_shift = abs((new.mean() - reference.mean()) / reference.std()) if reference.std() > 0 else 0
            
            # 4. Variance ratio test
            var_ratio = new.var() / reference.var() if reference.var() > 0 else 1.0
            variance_drift = abs(np.log(var_ratio))  # Log ratio for symmetric measure
            
            # 5. Distribution overlap (using histogram intersection)
            overlap_score = self._calculate_distribution_overlap(reference, new)
            
            # Combine scores for overall assessment
            drift_indicators = {
                'psi_score': psi_score,
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mean_shift': mean_shift,
                'variance_drift': variance_drift,
                'distribution_overlap': overlap_score
            }
            
            # Determine if drift is significant
            drift_detected = (
                psi_score > 0.2 or           # PSI threshold
                ks_p_value < 0.05 or         # KS test significance
                mean_shift > 2.0 or          # 2 standard deviations
                variance_drift > 0.5 or      # 50% variance change
                overlap_score < 0.7          # Less than 70% overlap
            )
            
            # Overall score (0-1, higher = more drift)
            overall_score = np.mean([
                min(psi_score / 0.3, 1.0),           # Normalize PSI
                min(ks_statistic, 1.0),              # KS statistic
                min(mean_shift / 3.0, 1.0),          # Normalize mean shift
                min(variance_drift / 0.7, 1.0),      # Normalize variance drift
                1.0 - overlap_score                   # Invert overlap (higher = more drift)
            ])
            
            return {
                'drift_detected': drift_detected,
                'overall_score': overall_score,
                'indicators': drift_indicators,
                'feature_name': feature_name,
                'reference_stats': {
                    'mean': reference.mean(),
                    'std': reference.std(),
                    'min': reference.min(),
                    'max': reference.max()
                },
                'new_stats': {
                    'mean': new.mean(),
                    'std': new.std(),
                    'min': new.min(),
                    'max': new.max()
                }
            }
            
        except Exception as e:
            logger.error(f"Numeric drift analysis failed for {feature_name}", error=str(e))
            return {'drift_detected': False, 'overall_score': 0.0, 'error': str(e)}
    
    def _analyze_categorical_feature_drift(self, reference: pd.Series, new: pd.Series, feature_name: str) -> Dict:
        """Analyze drift in categorical feature"""
        try:
            # 1. Chi-square test for distribution changes
            chi2_stat, chi2_p_value = self._chi_square_test(reference, new)
            
            # 2. Category frequency changes
            ref_freq = reference.value_counts(normalize=True)
            new_freq = new.value_counts(normalize=True)
            
            # Calculate frequency drift for common categories
            common_categories = set(ref_freq.index) & set(new_freq.index)
            freq_drifts = []
            
            for category in common_categories:
                freq_drift = abs(new_freq[category] - ref_freq[category])
                freq_drifts.append(freq_drift)
            
            avg_freq_drift = np.mean(freq_drifts) if freq_drifts else 0
            
            # 3. New/missing categories
            new_categories = set(new_freq.index) - set(ref_freq.index)
            missing_categories = set(ref_freq.index) - set(new_freq.index)
            
            # 4. Jensen-Shannon divergence
            js_divergence = self._calculate_js_divergence(ref_freq, new_freq)
            
            drift_indicators = {
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p_value,
                'avg_frequency_drift': avg_freq_drift,
                'js_divergence': js_divergence,
                'new_categories': list(new_categories),
                'missing_categories': list(missing_categories)
            }
            
            # Determine if drift is significant
            drift_detected = (
                chi2_p_value < 0.05 or       # Chi-square significance
                avg_freq_drift > 0.1 or      # Average frequency change > 10%
                len(new_categories) > 0 or   # New categories appeared
                len(missing_categories) > 0 or  # Categories disappeared
                js_divergence > 0.1          # JS divergence threshold
            )
            
            # Overall score
            overall_score = np.mean([
                min(avg_freq_drift * 5, 1.0),    # Normalize frequency drift
                min(js_divergence * 5, 1.0),     # Normalize JS divergence
                min(len(new_categories) / 3, 1.0),  # Normalize new categories
                min(len(missing_categories) / 3, 1.0)  # Normalize missing categories
            ])
            
            return {
                'drift_detected': drift_detected,
                'overall_score': overall_score,
                'indicators': drift_indicators,
                'feature_name': feature_name
            }
            
        except Exception as e:
            logger.error(f"Categorical drift analysis failed for {feature_name}", error=str(e))
            return {'drift_detected': False, 'overall_score': 0.0, 'error': str(e)}
    
    def _calculate_psi(self, reference: pd.Series, new: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            new_counts, _ = np.histogram(new, bins=bin_edges)
            
            # Convert to percentages (avoid division by zero)
            epsilon = 1e-6
            ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * buckets)
            new_pct = (new_counts + epsilon) / (len(new) + epsilon * buckets)
            
            # Calculate PSI
            psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
            
            return abs(psi)
            
        except Exception as e:
            logger.error("PSI calculation failed", error=str(e))
            return 0.0
    
    def _ks_test(self, reference: pd.Series, new: pd.Series) -> tuple:
        """Perform Kolmogorov-Smirnov test"""
        try:
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(reference, new)
            return statistic, p_value
        except ImportError:
            logger.warning("scipy not available for KS test")
            return 0.0, 1.0
        except Exception as e:
            logger.error("KS test failed", error=str(e))
            return 0.0, 1.0
    
    def _chi_square_test(self, reference: pd.Series, new: pd.Series) -> tuple:
        """Perform Chi-square test for categorical variables"""
        try:
            from scipy.stats import chi2_contingency
            
            # Get all unique categories
            all_categories = sorted(set(reference.unique()) | set(new.unique()))
            
            # Create frequency tables
            ref_counts = [sum(reference == cat) for cat in all_categories]
            new_counts = [sum(new == cat) for cat in all_categories]
            
            # Perform chi-square test
            contingency_table = np.array([ref_counts, new_counts])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            
            return chi2_stat, p_value
            
        except ImportError:
            logger.warning("scipy not available for chi-square test")
            return 0.0, 1.0
        except Exception as e:
            logger.error("Chi-square test failed", error=str(e))
            return 0.0, 1.0
    
    def _calculate_distribution_overlap(self, reference: pd.Series, new: pd.Series, bins: int = 20) -> float:
        """Calculate distribution overlap using histogram intersection"""
        try:
            # Use common range for both distributions
            min_val = min(reference.min(), new.min())
            max_val = max(reference.max(), new.max())
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate normalized histograms
            ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
            new_hist, _ = np.histogram(new, bins=bin_edges, density=True)
            
            # Normalize to get probabilities
            ref_hist = ref_hist / np.sum(ref_hist)
            new_hist = new_hist / np.sum(new_hist)
            
            # Calculate overlap (intersection)
            overlap = np.sum(np.minimum(ref_hist, new_hist))
            
            return overlap
            
        except Exception as e:
            logger.error("Distribution overlap calculation failed", error=str(e))
            return 0.0
    
    def _calculate_js_divergence(self, ref_freq: pd.Series, new_freq: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence for categorical distributions"""
        try:
            # Get all categories
            all_categories = sorted(set(ref_freq.index) | set(new_freq.index))
            
            # Create aligned probability vectors
            ref_probs = np.array([ref_freq.get(cat, 0) for cat in all_categories])
            new_probs = np.array([new_freq.get(cat, 0) for cat in all_categories])
            
            # Normalize
            ref_probs = ref_probs / np.sum(ref_probs) if np.sum(ref_probs) > 0 else ref_probs
            new_probs = new_probs / np.sum(new_probs) if np.sum(new_probs) > 0 else new_probs
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_probs = ref_probs + epsilon
            new_probs = new_probs + epsilon
            
            # Calculate JS divergence
            m = (ref_probs + new_probs) / 2
            js_div = 0.5 * np.sum(ref_probs * np.log(ref_probs / m)) + 0.5 * np.sum(new_probs * np.log(new_probs / m))
            
            return js_div
            
        except Exception as e:
            logger.error("JS divergence calculation failed", error=str(e))
            return 0.0
    
    def validate_survey_data(self, data: Dict) -> ValidationResult:
        """Validate individual survey data submission"""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame([data])
            
            # Schema validation
            schema_errors = self._validate_schema(df)
            errors.extend(schema_errors)
            
            # Data quality checks
            quality_errors, quality_warnings = self._validate_data_quality(df)
            errors.extend(quality_errors)
            warnings.extend(quality_warnings)
            
            # Business rule validation
            business_errors, business_warnings = self._validate_business_rules(df)
            errors.extend(business_errors)
            warnings.extend(business_warnings)
            
            # Statistical validation
            stat_warnings = self._validate_statistical_consistency(df)
            warnings.extend(stat_warnings)
            
            # Generate metrics
            metrics = self._generate_validation_metrics(df, errors, warnings)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                validated_records=1,
                failed_records=1 if errors else 0
            )
            
        except Exception as e:
            logger.error("Validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation system error: {str(e)}"],
                warnings=[],
                metrics={},
                validated_records=0,
                failed_records=1
            )
    
    def validate_batch_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate batch data upload for 75-variable structure"""
        errors = []
        warnings = []
        total_records = len(df)
        failed_records = 0
        
        try:
            logger.info("Starting batch data validation", total_records=total_records)
            
            # Schema validation for new dataset structure
            schema_errors = self._validate_batch_schema(df)
            errors.extend(schema_errors)
            
            # Data quality validation with higher standards
            quality_errors, quality_warnings, quality_failed = self._validate_batch_quality(df)
            errors.extend(quality_errors)
            warnings.extend(quality_warnings)
            failed_records += quality_failed
            
            # Business rule validation with new thresholds
            business_errors, business_warnings, business_failed = self._validate_batch_business_rules(df)
            errors.extend(business_errors)
            warnings.extend(business_warnings)
            failed_records += business_failed
            
            # Statistical validation for the batch
            stat_warnings = self._validate_batch_statistics(df)
            warnings.extend(stat_warnings)
            
            # Generate comprehensive metrics
            metrics = self._generate_batch_metrics(df, errors, warnings, failed_records)
            
            return ValidationResult(
                is_valid=failed_records < total_records * 0.05,  # Allow up to 5% failures (higher standard)
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                validated_records=total_records,
                failed_records=failed_records
            )
            
        except Exception as e:
            logger.error("Batch validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                errors=[f"Batch validation system error: {str(e)}"],
                warnings=[],
                metrics={},
                validated_records=total_records,
                failed_records=total_records
            )
    
    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate data schema for new 75-variable structure"""
        errors = []
        
        # Essential columns from the new dataset
        essential_columns = config.ml.essential_features
        
        missing_columns = [col for col in essential_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing essential columns: {missing_columns}")
        
        # Data type validation for key columns
        if 'HouseholdSize' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['HouseholdSize']):
                try:
                    pd.to_numeric(df['HouseholdSize'], errors='raise')
                except:
                    errors.append("HouseholdSize must be numeric")
        
        if 'HHIncome+Consumption+Residues/Day' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['HHIncome+Consumption+Residues/Day']):
                try:
                    pd.to_numeric(df['HHIncome+Consumption+Residues/Day'], errors='raise')
                except:
                    errors.append("Target variable must be numeric")
        
        # Validate geographic columns
        if 'District' in df.columns:
            valid_districts = ['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo']  # From EDA
            invalid_districts = df[~df['District'].isin(valid_districts)]['District'].unique()
            if len(invalid_districts) > 0:
                errors.append(f"Invalid districts found: {invalid_districts.tolist()}")
        
        return errors
    
    def _validate_batch_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate schema for batch data with 75-variable structure"""
        errors = []
        
        # Check expected column count
        expected_columns = 75
        if len(df.columns) != expected_columns:
            warnings.append(f"Expected {expected_columns} columns, found {len(df.columns)}")
        
        # Use Great Expectations for comprehensive schema validation
        ge_df = ge.from_pandas(df)
        
        # Essential columns must exist
        essential_columns = config.ml.essential_features
        for col in essential_columns:
            if col not in df.columns:
                errors.append(f"Essential column missing: {col}")
        
        # Validate geographic hierarchy
        geographic_cols = ['District', 'Cluster', 'Village']
        for col in geographic_cols:
            if col in df.columns:
                if df[col].isnull().all():
                    errors.append(f"Geographic column {col} is completely empty")
        
        # Validate target variable
        target_col = config.ml.target_variable
        if target_col in df.columns:
            try:
                target_values = pd.to_numeric(df[target_col], errors='coerce')
                if target_values.isnull().all():
                    errors.append(f"Target variable {target_col} contains no valid numeric values")
            except:
                errors.append(f"Target variable {target_col} validation failed")
        
        return errors
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data quality for individual records with high standards"""
        errors = []
        warnings = []
        
        # Check for missing essential values
        essential_cols = ['HouseholdSize', 'District', 'HHIncome+Consumption+Residues/Day']
        for col in essential_cols:
            if col in df.columns and df[col].isnull().any():
                errors.append(f"Missing value in essential column: {col}")
        
        # Validate household size
        if 'HouseholdSize' in df.columns:
            household_size = df['HouseholdSize'].iloc[0] if len(df) > 0 else None
            if household_size is not None:
                if household_size <= 0 or household_size > 20:
                    errors.append(f"Invalid household size: {household_size}")
        
        # Validate target variable range
        target_col = 'HHIncome+Consumption+Residues/Day'
        if target_col in df.columns:
            income = df[target_col].iloc[0] if len(df) > 0 else None
            if income is not None:
                if income < 0:
                    errors.append(f"Negative income value: {income}")
                elif income > 50:  # Very high income threshold
                    warnings.append(f"Unusually high income: {income}")
        
        # Validate agricultural data consistency
        agric_cols = ['AgricultureLand', 'Season1CropsPlanted', 'Season2CropsPlanted']
        agric_values = {}
        for col in agric_cols:
            if col in df.columns:
                agric_values[col] = df[col].iloc[0] if len(df) > 0 else None
        
        # Check agricultural consistency
        if agric_values.get('AgricultureLand', 0) == 0:
            if any(agric_values.get(col, 0) > 0 for col in ['Season1CropsPlanted', 'Season2CropsPlanted']):
                warnings.append("Crops planted but no agricultural land reported")
        
        return errors, warnings
    
    def _validate_batch_quality(self, df: pd.DataFrame) -> Tuple[List[str], List[str], int]:
        """Validate batch data quality with high standards for 97.4% completeness expectation"""
        errors = []
        warnings = []
        failed_records = 0
        
        # Overall completeness check - should be near 97.4%
        completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
        if completeness < config.pipeline.completeness_threshold:
            warnings.append(f"Data completeness {completeness:.1%} below expected {config.pipeline.completeness_threshold:.1%}")
        
        # Essential columns completeness
        essential_cols = config.ml.essential_features
        for col in essential_cols:
            if col in df.columns:
                missing_pct = df[col].isnull().mean()
                if missing_pct > 0.05:  # More than 5% missing
                    errors.append(f"High missing rate in {col}: {missing_pct:.1%}")
                    failed_records += df[col].isnull().sum()
        
        # Household size validation
        if 'HouseholdSize' in df.columns:
            invalid_size = df[(df['HouseholdSize'] <= 0) | (df['HouseholdSize'] > 20)]
            if len(invalid_size) > 0:
                errors.append(f"Invalid household sizes: {len(invalid_size)} records")
                failed_records += len(invalid_size)
        
        # Income validation
        target_col = config.ml.target_variable
        if target_col in df.columns:
            negative_income = df[df[target_col] < 0]
            if len(negative_income) > 0:
                errors.append(f"Negative income values: {len(negative_income)} records")
                failed_records += len(negative_income)
            
            extreme_income = df[df[target_col] > 50]
            if len(extreme_income) > 0:
                warnings.append(f"Extremely high income values: {len(extreme_income)} records")
        
        # Geographic data validation
        if 'District' in df.columns:
            valid_districts = ['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo']
            invalid_districts = df[~df['District'].isin(valid_districts)]
            if len(invalid_districts) > 0:
                errors.append(f"Invalid district values: {len(invalid_districts)} records")
                failed_records += len(invalid_districts)
        
        # Duplicate household check
        if 'HouseHoldID' in df.columns:
            duplicates = df[df['HouseHoldID'].duplicated()]
            if len(duplicates) > 0:
                errors.append(f"Duplicate household IDs: {len(duplicates)} records")
                failed_records += len(duplicates)
        
        return errors, warnings, failed_records
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate business rules for household vulnerability assessment"""
        errors = []
        warnings = []
        
        # Vulnerability classification consistency
        target_col = 'HHIncome+Consumption+Residues/Day'
        if target_col in df.columns:
            income = df[target_col].iloc[0] if len(df) > 0 else None
            if income is not None:
                vulnerability_class = config.classify_vulnerability(income)
                
                # Check consistency with other indicators
                if 'VSLA_Profits' in df.columns:
                    vsla_profits = df['VSLA_Profits'].iloc[0]
                    if vulnerability_class in ['Severely Struggling', 'Struggling'] and vsla_profits > 1000:
                        warnings.append("High VSLA profits inconsistent with struggling status")
                
                if 'BusinessIncome' in df.columns:
                    business_income = df['BusinessIncome'].iloc[0]
                    if vulnerability_class == 'On Track' and business_income == 0:
                        warnings.append("No business income but classified as On Track")
        
        return errors, warnings
    
    def _validate_batch_business_rules(self, df: pd.DataFrame) -> Tuple[List[str], List[str], int]:
        """Validate business rules for batch data"""
        errors = []
        warnings = []
        failed_records = 0
        
        target_col = config.ml.target_variable
        if target_col in df.columns:
            # Apply vulnerability classification
            df_temp = df.copy()
            df_temp['vulnerability_class'] = df_temp[target_col].apply(config.classify_vulnerability)
            
            # Check vulnerability distribution
            vuln_dist = df_temp['vulnerability_class'].value_counts(normalize=True)
            
            # Expected distribution based on Part A analysis
            expected_struggling = 0.425  # 42.5% vulnerable
            actual_struggling = vuln_dist.get('Struggling', 0) + vuln_dist.get('Severely Struggling', 0)
            
            if abs(actual_struggling - expected_struggling) > 0.1:  # 10% tolerance
                warnings.append(f"Vulnerability distribution differs from expected: {actual_struggling:.1%} vs {expected_struggling:.1%}")
        
        return errors, warnings, failed_records
    
    def _validate_statistical_consistency(self, df: pd.DataFrame) -> List[str]:
        """Validate statistical consistency for individual records"""
        warnings = []
        
        # Agricultural land vs income consistency
        if all(col in df.columns for col in ['AgricultureLand', 'HHIncome+Consumption+Residues/Day']):
            land = df['AgricultureLand'].iloc[0] if len(df) > 0 else None
            income = df['HHIncome+Consumption+Residues/Day'].iloc[0] if len(df) > 0 else None
            
            if land is not None and income is not None:
                if land > 5 and income < 1.0:  # Large land but low income
                    warnings.append("Large agricultural land but low income - potential data inconsistency")
        
        return warnings
    
    def _validate_batch_statistics(self, df: pd.DataFrame) -> List[str]:
        """Validate statistical consistency for batch data"""
        warnings = []
        
        # Check expected ranges based on Part A analysis
        if 'HouseholdSize' in df.columns:
            avg_household_size = df['HouseholdSize'].mean()
            expected_size = 4.8  # From Part A analysis
            if abs(avg_household_size - expected_size) > 1.0:
                warnings.append(f"Average household size {avg_household_size:.1f} differs from expected {expected_size}")
        
        if 'HHIncome+Consumption+Residues/Day' in df.columns:
            avg_income = df['HHIncome+Consumption+Residues/Day'].mean()
            expected_income = 2.24  # From Part A analysis
            if abs(avg_income - expected_income) > 0.5:
                warnings.append(f"Average income {avg_income:.2f} differs from expected {expected_income}")
        
        return warnings
    
    def _generate_validation_metrics(self, df: pd.DataFrame, errors: List[str], warnings: List[str]) -> Dict:
        """Generate validation metrics for individual records"""
        return {
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "completeness_rate": 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_batch_metrics(self, df: pd.DataFrame, errors: List[str], 
                              warnings: List[str], failed_records: int) -> Dict:
        """Generate comprehensive batch validation metrics"""
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "completeness_rate": 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "failed_records": failed_records,
            "success_rate": (len(df) - failed_records) / len(df),
            "validation_timestamp": datetime.utcnow().isoformat(),
            "dataset_info": config.dataset_info
        }
    
    def _create_expectations_suite(self) -> ExpectationSuite:
        """Create Great Expectations suite for the 75-variable dataset"""
        suite = ExpectationSuite(expectation_suite_name="household_survey_expectations_v2")
        
        # Add expectations for key columns
        essential_features = config.ml.essential_features
        for feature in essential_features:
            if feature in ['HouseholdSize']:
                suite.add_expectation({
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"column": feature, "min_value": 1, "max_value": 20}
                })
        
        return suite
    
    def _load_business_rules(self) -> Dict:
        """Load business rules for the assessment"""
        return {
            "vulnerability_thresholds": config.ml.target_thresholds,
            "max_household_size": 20,
            "max_reasonable_income": 50.0,
            "min_income": 0.0
        }
    
    def _load_data_dictionary(self) -> Dict:
        """Load data dictionary for the 75-variable dataset"""
        return {
            "feature_categories": config.ml.feature_categories,
            "essential_features": config.ml.essential_features,
            "target_variable": config.ml.target_variable,
            "total_variables": config.dataset_info["total_variables"]
        }

def calculate_psi(expected, actual, buckets=10):
    expected_pct = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=buckets)[0] / len(actual)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi 

def calculate_jensenshannon(reference_dist, new_dist):
    js_distance = jensenshannon(reference_dist, new_dist)
    drift_detected = js_distance > config.ml.data_drift_threshold
    return drift_detected 