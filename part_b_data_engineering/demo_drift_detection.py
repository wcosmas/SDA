#!/usr/bin/env python3
"""
Enhanced Data Drift Detection Demonstration
Part B: Data Engineering Pipeline

This script demonstrates the statistical drift detection methods
now implemented in the pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from validation.data_validator import DataValidator, DriftAnalysisResult
from pipeline.etl_orchestrator import ETLOrchestrator


def create_sample_reference_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create reference dataset for drift detection demo"""
    np.random.seed(42)  # For reproducible results
    
    return pd.DataFrame({
        'HouseholdSize': np.random.poisson(5, n_samples),
        'AgricultureLand': np.random.gamma(2, 1.5, n_samples),
        'HHIncome+Consumption+Residues/Day': np.random.lognormal(1.0, 0.8, n_samples),
        'VSLA_Profits': np.random.exponential(1000, n_samples),
        'BusinessIncome': np.random.gamma(1.5, 2000, n_samples),
        'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
        'TimeToWater': np.random.gamma(1.0, 10, n_samples),
        'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo'], 
                                   n_samples, p=[0.3, 0.25, 0.25, 0.2])
    })


def create_drifted_data(reference_data: pd.DataFrame, drift_type: str = "none") -> pd.DataFrame:
    """Create new data with different types of drift"""
    n_samples = len(reference_data) // 2
    np.random.seed(123)  # Different seed for new data
    
    if drift_type == "none":
        # No drift - same distribution
        return pd.DataFrame({
            'HouseholdSize': np.random.poisson(5, n_samples),
            'AgricultureLand': np.random.gamma(2, 1.5, n_samples),
            'HHIncome+Consumption+Residues/Day': np.random.lognormal(1.0, 0.8, n_samples),
            'VSLA_Profits': np.random.exponential(1000, n_samples),
            'BusinessIncome': np.random.gamma(1.5, 2000, n_samples),
            'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
            'TimeToWater': np.random.gamma(1.0, 10, n_samples),
            'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo'], 
                                       n_samples, p=[0.3, 0.25, 0.25, 0.2])
        })
    
    elif drift_type == "mean_shift":
        # Mean shift in key variables
        return pd.DataFrame({
            'HouseholdSize': np.random.poisson(6.5, n_samples),  # Increased mean
            'AgricultureLand': np.random.gamma(2, 2.0, n_samples),  # Increased scale
            'HHIncome+Consumption+Residues/Day': np.random.lognormal(1.3, 0.8, n_samples),  # Higher income
            'VSLA_Profits': np.random.exponential(1500, n_samples),  # Higher profits
            'BusinessIncome': np.random.gamma(1.5, 2000, n_samples),
            'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
            'TimeToWater': np.random.gamma(1.0, 10, n_samples),
            'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo'], 
                                       n_samples, p=[0.3, 0.25, 0.25, 0.2])
        })
    
    elif drift_type == "variance_change":
        # Variance change
        return pd.DataFrame({
            'HouseholdSize': np.random.poisson(5, n_samples),
            'AgricultureLand': np.random.gamma(2, 3.0, n_samples),  # Increased variance
            'HHIncome+Consumption+Residues/Day': np.random.lognormal(1.0, 1.5, n_samples),  # Higher variance
            'VSLA_Profits': np.random.exponential(1000, n_samples),
            'BusinessIncome': np.random.gamma(1.5, 4000, n_samples),  # Higher variance
            'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
            'TimeToWater': np.random.gamma(1.0, 10, n_samples),
            'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo'], 
                                       n_samples, p=[0.3, 0.25, 0.25, 0.2])
        })
    
    elif drift_type == "distribution_shift":
        # Complete distribution change
        return pd.DataFrame({
            'HouseholdSize': np.random.negative_binomial(3, 0.4, n_samples),  # Different distribution
            'AgricultureLand': np.random.weibull(1.5, n_samples) * 2,  # Different distribution
            'HHIncome+Consumption+Residues/Day': np.random.gamma(2, 800, n_samples),  # Different distribution
            'VSLA_Profits': np.random.gamma(2, 500, n_samples),  # Different distribution
            'BusinessIncome': np.random.gamma(1.5, 2000, n_samples),
            'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
            'TimeToWater': np.random.gamma(1.0, 10, n_samples),
            'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo'], 
                                       n_samples, p=[0.15, 0.15, 0.35, 0.35])  # Different distribution
        })
    
    elif drift_type == "new_categories":
        # New categorical values
        return pd.DataFrame({
            'HouseholdSize': np.random.poisson(5, n_samples),
            'AgricultureLand': np.random.gamma(2, 1.5, n_samples),
            'HHIncome+Consumption+Residues/Day': np.random.lognormal(1.0, 0.8, n_samples),
            'VSLA_Profits': np.random.exponential(1000, n_samples),
            'BusinessIncome': np.random.gamma(1.5, 2000, n_samples),
            'TimeToOPD': np.random.gamma(1.2, 15, n_samples),
            'TimeToWater': np.random.gamma(1.0, 10, n_samples),
            'District': np.random.choice(['Mitooma', 'Kanungu', 'Rubirizi', 'Ntungamo', 'Kabale'], 
                                       n_samples, p=[0.25, 0.2, 0.2, 0.15, 0.2])  # New district added
        })


def demonstrate_drift_detection():
    """Demonstrate different types of drift detection"""
    print("🔍 ENHANCED DATA DRIFT DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create reference data
    print("\n1. Creating reference dataset...")
    reference_data = create_sample_reference_data(1000)
    print(f"   Reference data shape: {reference_data.shape}")
    print(f"   Reference data summary:")
    print(f"   - Mean HouseholdSize: {reference_data['HouseholdSize'].mean():.2f}")
    print(f"   - Mean Income/Day: {reference_data['HHIncome+Consumption+Residues/Day'].mean():.2f}")
    print(f"   - District distribution: {reference_data['District'].value_counts().to_dict()}")
    
    # Initialize validator
    validator = DataValidator()
    validator.set_reference_data(reference_data)
    
    # Test different drift scenarios
    drift_scenarios = [
        ("none", "No Drift"),
        ("mean_shift", "Mean Shift Drift"),
        ("variance_change", "Variance Change Drift"),
        ("distribution_shift", "Distribution Shift Drift"),
        ("new_categories", "New Categories Drift")
    ]
    
    results = {}
    
    for drift_type, drift_name in drift_scenarios:
        print(f"\n2. Testing {drift_name}...")
        print("-" * 40)
        
        # Create drifted data
        new_data = create_drifted_data(reference_data, drift_type)
        print(f"   New data shape: {new_data.shape}")
        
        if drift_type == "mean_shift":
            print(f"   - Mean HouseholdSize: {new_data['HouseholdSize'].mean():.2f} (vs {reference_data['HouseholdSize'].mean():.2f})")
            print(f"   - Mean Income/Day: {new_data['HHIncome+Consumption+Residues/Day'].mean():.2f} (vs {reference_data['HHIncome+Consumption+Residues/Day'].mean():.2f})")
        elif drift_type == "new_categories":
            print(f"   - District distribution: {new_data['District'].value_counts().to_dict()}")
        
        # Perform drift analysis
        drift_result = validator.analyze_data_drift(new_data)
        results[drift_name] = drift_result
        
        # Display results
        print(f"\n   📊 Drift Analysis Results:")
        print(f"   - Drift Detected: {'🚨 YES' if drift_result.drift_detected else '✅ NO'}")
        print(f"   - Overall Drift Score: {drift_result.overall_drift_score:.4f}")
        print(f"   - Features Analyzed: {len(drift_result.feature_drift_scores)}")
        print(f"   - Significant Features: {len(drift_result.significant_features)}")
        
        if drift_result.significant_features:
            print(f"   - Drifted Features: {', '.join(drift_result.significant_features)}")
        
        # Show top feature drift scores
        if drift_result.feature_drift_scores:
            top_features = sorted(drift_result.feature_drift_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print(f"   - Top Drift Scores:")
            for feature, score in top_features:
                print(f"     • {feature}: {score:.4f}")
    
    # Summary
    print(f"\n3. DRIFT DETECTION SUMMARY")
    print("=" * 60)
    for scenario_name, result in results.items():
        status = "DRIFT DETECTED" if result.drift_detected else "NO DRIFT"
        print(f"{scenario_name:<25} | {status:<15} | Score: {result.overall_drift_score:.4f}")
    
    return results


def demonstrate_orchestrator_drift_detection():
    """Demonstrate drift detection in ETL orchestrator"""
    print(f"\n4. ETL ORCHESTRATOR DRIFT DETECTION")
    print("-" * 50)
    
    # Initialize orchestrator
    orchestrator = ETLOrchestrator()
    
    # Test PSI calculation
    reference_series = pd.Series(np.random.normal(100, 15, 1000))
    new_series_no_drift = pd.Series(np.random.normal(100, 15, 500))
    new_series_with_drift = pd.Series(np.random.normal(120, 20, 500))  # Mean and variance shift
    
    psi_no_drift = orchestrator._calculate_psi(reference_series, new_series_no_drift)
    psi_with_drift = orchestrator._calculate_psi(reference_series, new_series_with_drift)
    
    print(f"PSI Score (No Drift): {psi_no_drift:.4f}")
    print(f"PSI Score (With Drift): {psi_with_drift:.4f}")
    print(f"PSI Threshold: 0.2 (Higher values indicate more drift)")
    
    # Test KS test
    ks_stat_no_drift, ks_p_no_drift = orchestrator._kolmogorov_smirnov_test(reference_series, new_series_no_drift)
    ks_stat_with_drift, ks_p_with_drift = orchestrator._kolmogorov_smirnov_test(reference_series, new_series_with_drift)
    
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"No Drift - Statistic: {ks_stat_no_drift:.4f}, p-value: {ks_p_no_drift:.4f}")
    print(f"With Drift - Statistic: {ks_stat_with_drift:.4f}, p-value: {ks_p_with_drift:.4f}")
    print(f"Significance threshold: p < 0.05")
    
    # Test categorical drift
    reference_cat = pd.Series(['A', 'B', 'C'] * 300 + ['D'] * 100)
    new_cat_no_drift = pd.Series(['A', 'B', 'C'] * 150 + ['D'] * 50)
    new_cat_with_drift = pd.Series(['A'] * 100 + ['B'] * 50 + ['C'] * 200 + ['E'] * 50)  # Different distribution + new category
    
    chi2_no_drift, chi2_p_no_drift = orchestrator._chi_square_test(reference_cat, new_cat_no_drift)
    chi2_with_drift, chi2_p_with_drift = orchestrator._chi_square_test(reference_cat, new_cat_with_drift)
    
    print(f"\nChi-Square Test (Categorical):")
    print(f"No Drift - Statistic: {chi2_no_drift:.4f}, p-value: {chi2_p_no_drift:.4f}")
    print(f"With Drift - Statistic: {chi2_with_drift:.4f}, p-value: {chi2_p_with_drift:.4f}")


if __name__ == "__main__":
    try:
        # Run drift detection demonstration
        drift_results = demonstrate_drift_detection()
        
        # Run orchestrator-level demonstration
        demonstrate_orchestrator_drift_detection()
        
        print(f"\n✅ DRIFT DETECTION DEMONSTRATION COMPLETED")
        print(f"The pipeline now uses statistical methods including:")
        print(f"• Population Stability Index (PSI)")
        print(f"• Kolmogorov-Smirnov tests")
        print(f"• Chi-square tests for categorical variables")
        print(f"• Mean shift detection")
        print(f"• Variance change detection")
        print(f"• Distribution overlap analysis")
        print(f"• Jensen-Shannon divergence")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc() 