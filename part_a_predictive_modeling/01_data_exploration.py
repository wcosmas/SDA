#!/usr/bin/env python3
"""
RTV Senior Data Scientist Technical Assessment - Part A
01. Data Exploration Script

This script performs initial exploration of the DataScientist_01_Assessment.csv dataset
to understand the structure, quality, and characteristics of the household survey data.

Author: Cosmas Wamozo
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    
    print("=" * 80)
    print("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - DATA EXPLORATION")
    print("=" * 80)
    
    # Load the dataset
    try:
        print("\n1. LOADING DATASET...")
        df = pd.read_csv('../DataScientist_01_Assessment.csv')
        print(f"✓ Successfully loaded dataset: {df.shape[0]} households, {df.shape[1]} variables")
        
        # Load data dictionary
        dict_df = pd.read_excel('../Dictionary.xlsx')
        print(f"✓ Successfully loaded data dictionary: {len(dict_df)} variable definitions")
        
    except FileNotFoundError as e:
        print(f"✗ Error loading files: {e}")
        return None, None
    
    print("\n2. DATASET OVERVIEW...")
    print(f"Dataset dimensions: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types analysis
    print("\n3. DATA TYPES ANALYSIS...")
    dtype_summary = df.dtypes.value_counts()
    print("Data types distribution:")
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values analysis
    print("\n4. MISSING VALUES ANALYSIS...")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percentage.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(f"Columns with missing data: {sum(missing_data > 0)}")
    
    # Show columns with significant missing data
    significant_missing = missing_summary[missing_summary['Missing_Percentage'] > 0]
    if not significant_missing.empty:
        print("\nColumns with missing values:")
        for _, row in significant_missing.head(10).iterrows():
            print(f"  {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
    else:
        print("✓ No significant missing values found!")
    
    # Geographic distribution
    print("\n5. GEOGRAPHIC DISTRIBUTION...")
    print("Districts:")
    district_counts = df['District'].value_counts()
    for district, count in district_counts.items():
        print(f"  {district}: {count} households ({count/len(df)*100:.1f}%)")
    
    print(f"\nTotal clusters: {df['Cluster'].nunique()}")
    print(f"Total villages: {df['Village'].nunique()}")
    
    # Household characteristics
    print("\n6. HOUSEHOLD CHARACTERISTICS...")
    print(f"Household size - Mean: {df['HouseholdSize'].mean():.1f}, Range: {df['HouseholdSize'].min()}-{df['HouseholdSize'].max()}")
    
    # Infrastructure access
    print("\n7. INFRASTRUCTURE ACCESS...")
    print(f"Time to OPD - Mean: {df['TimeToOPD'].mean():.1f} minutes")
    print(f"Time to Water - Mean: {df['TimeToWater'].mean():.1f} minutes")
    print(f"Agriculture Land - Mean: {df['AgricultureLand'].mean():.2f} acres")
    
    # Income components analysis
    print("\n8. INCOME COMPONENTS ANALYSIS...")
    income_cols = [
        'Season1VegetableIncome', 'Season2VegatableIncome', 'VegetableIncome',
        'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 'CasualLabour',
        'RemittancesAndGifts', 'RentIncome', 'SeasonalCropIncome',
        'PerenialCropIncome', 'LivestockIncome'
    ]
    
    print("Average income by source (annual):")
    for col in income_cols:
        if col in df.columns:
            print(f"  {col}: ${df[col].mean():.2f}")
    
    # Daily income analysis
    print("\n9. DAILY INCOME ANALYSIS...")
    daily_income_cols = [
        'HHIncome/Day', 'Consumption/Day', 
        'HHIncome+Consumption+Residues/Day',
        'HHIncome+Consumption+Assets+Residues/Day'
    ]
    
    for col in daily_income_cols:
        if col in df.columns:
            print(f"{col}:")
            print(f"  Mean: ${df[col].mean():.2f}/day")
            print(f"  Median: ${df[col].median():.2f}/day") 
            print(f"  Range: ${df[col].min():.2f} - ${df[col].max():.2f}/day")
    
    # Asset and infrastructure indicators
    print("\n10. ASSET AND INFRASTRUCTURE INDICATORS...")
    binary_indicators = [
        'radios_owned', 'phones_owned', 'work_casual', 'work_salaried',
        'latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
        'standard_hangline', 'kitchen_house', 'bathroom_constructed'
    ]
    
    print("Infrastructure and asset ownership rates:")
    for col in binary_indicators:
        if col in df.columns:
            rate = (df[col] == 1).mean() * 100
            print(f"  {col}: {rate:.1f}%")
    
    return df, dict_df

def create_data_summary(df, dict_df):
    """Create and save data summary"""
    
    print("\n11. CREATING DATA SUMMARY...")
    
    # Create comprehensive summary
    summary_data = []
    
    for col in df.columns:
        # Get description from dictionary
        description = "No description available"
        dict_match = dict_df[dict_df['Variable'] == col]
        if not dict_match.empty:
            description = dict_match['Description'].iloc[0]
        
        # Calculate statistics
        if df[col].dtype in ['int64', 'float64']:
            summary_data.append({
                'Variable': col,
                'Description': description,
                'Type': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percentage': (df[col].isnull().sum() / len(df)) * 100,
                'Mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'Std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'Min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'Max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'Unique_Values': df[col].nunique()
            })
        else:
            summary_data.append({
                'Variable': col,
                'Description': description,
                'Type': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percentage': (df[col].isnull().sum() / len(df)) * 100,
                'Mean': None,
                'Std': None,
                'Min': None,
                'Max': None,
                'Unique_Values': df[col].nunique()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv('column_information.csv', index=False)
    print("✓ Saved detailed column information to 'column_information.csv'")
    
    # Create basic statistics summary
    basic_stats = {
        'total_households': len(df),
        'total_variables': len(df.columns),
        'numeric_variables': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_variables': len(df.select_dtypes(include=['object']).columns),
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'districts_covered': df['District'].nunique(),
        'villages_covered': df['Village'].nunique(),
        'avg_household_size': df['HouseholdSize'].mean(),
        'avg_daily_income': df['HHIncome/Day'].mean(),
        'avg_daily_total': df['HHIncome+Consumption+Residues/Day'].mean()
    }
    
    print("\nDATASET SUMMARY STATISTICS:")
    print("=" * 50)
    for key, value in basic_stats.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    return summary_df

def main():
    """Main execution function"""
    print("Starting data exploration...")
    
    # Load and explore data
    df, dict_df = load_and_explore_data()
    
    if df is not None:
        # Create summary
        summary_df = create_data_summary(df, dict_df)
        
        print("\n" + "=" * 80)
        print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Findings:")
        print(f"• Dataset contains {len(df):,} households with {len(df.columns)} variables")
        print(f"• Data covers {df['District'].nunique()} districts and {df['Village'].nunique()} villages")
        print(f"• Average household size: {df['HouseholdSize'].mean():.1f} members")
        print(f"• Average daily total income: ${df['HHIncome+Consumption+Residues/Day'].mean():.2f}")
        print(f"• Data quality: {((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}% complete")
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"• Columns with missing data: {len(missing_cols)}")
        else:
            print("• ✓ No missing data!")
        
        print("\nNext steps:")
        print("1. Run '02_target_variable_creation.py' to create vulnerability status")
        print("2. Run '03_comprehensive_eda.py' for detailed analysis")
        print("3. Proceed with modeling pipeline")
        
    else:
        print("✗ Data exploration failed. Please check file paths and data integrity.")

if __name__ == "__main__":
    main()