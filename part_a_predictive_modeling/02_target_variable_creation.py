#!/usr/bin/env python3
"""
RTV Senior Data Scientist Technical Assessment - Part A
02. Target Variable Creation Script

This script creates the ProgressStatus target variable from the 
HHIncome+Consumption+Residues/Day column according to assessment requirements.

Author: Cosmas Wamozo
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the dataset and data dictionary"""
    
    print("=" * 80)
    print("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - TARGET VARIABLE CREATION")
    print("=" * 80)
    
    try:
        print("\n1. LOADING DATASET...")
        df = pd.read_csv('../DataScientist_01_Assessment.csv')
        print(f"✓ Successfully loaded dataset: {df.shape[0]} households, {df.shape[1]} variables")
        
        # Load data dictionary
        dict_df = pd.read_excel('../Dictionary.xlsx')
        print(f"✓ Successfully loaded data dictionary: {len(dict_df)} variable definitions")
        
        return df, dict_df
        
    except FileNotFoundError as e:
        print(f"✗ Error loading files: {e}")
        return None, None

def create_progress_status(df):
    """Create ProgressStatus variable according to assessment requirements"""
    
    print("\n2. CREATING PROGRESS STATUS TARGET VARIABLE...")
    
    # Target variable as specified in assessment
    target_col = 'HHIncome+Consumption+Residues/Day'
    
    if target_col not in df.columns:
        print(f"✗ Target column '{target_col}' not found in dataset")
        return None
    
    print(f"✓ Using target column: {target_col}")
    
    # Verify data quality
    print(f"\nTarget variable data quality:")
    print(f"  Total records: {len(df)}")
    print(f"  Non-null records: {df[target_col].count()}")
    print(f"  Missing records: {df[target_col].isnull().sum()}")
    print(f"  Data completeness: {(df[target_col].count() / len(df)) * 100:.1f}%")
    
    # Basic statistics
    print(f"\nTarget variable statistics:")
    print(f"  Mean: ${df[target_col].mean():.2f}/day")
    print(f"  Median: ${df[target_col].median():.2f}/day")
    print(f"  Standard deviation: ${df[target_col].std():.2f}/day")
    print(f"  Minimum: ${df[target_col].min():.2f}/day")
    print(f"  Maximum: ${df[target_col].max():.2f}/day")
    
    # Create ProgressStatus according to assessment criteria
    print(f"\n3. APPLYING PROGRESS STATUS CRITERIA...")
    print("Thresholds (as per assessment requirements):")
    print("  • On Track: >= $2.15/day")
    print("  • At Risk: >= $1.77/day and < $2.15/day")
    print("  • Struggling: >= $1.25/day and < $1.77/day")
    print("  • Severely Struggling: < $1.25/day")
    
    def assign_progress_status(income_per_day):
        """Assign progress status based on daily income"""
        if pd.isna(income_per_day):
            return 'Unknown'
        elif income_per_day >= 2.15:
            return 'On Track'
        elif income_per_day >= 1.77:
            return 'At Risk'
        elif income_per_day >= 1.25:
            return 'Struggling'
        else:
            return 'Severely Struggling'
    
    # Apply the function
    df['ProgressStatus'] = df[target_col].apply(assign_progress_status)
    
    # Analyze the distribution
    print(f"\n4. PROGRESS STATUS DISTRIBUTION ANALYSIS...")
    
    status_counts = df['ProgressStatus'].value_counts()
    status_percentages = df['ProgressStatus'].value_counts(normalize=True) * 100
    
    print("Distribution of ProgressStatus:")
    for status in ['On Track', 'At Risk', 'Struggling', 'Severely Struggling', 'Unknown']:
        if status in status_counts.index:
            count = status_counts[status]
            percentage = status_percentages[status]
            print(f"  {status}: {count:,} households ({percentage:.1f}%)")
    
    # Calculate vulnerability statistics
    vulnerable_statuses = ['Struggling', 'Severely Struggling']
    vulnerable_count = df[df['ProgressStatus'].isin(vulnerable_statuses)].shape[0]
    vulnerable_percentage = (vulnerable_count / len(df)) * 100
    
    print(f"\nVulnerability Summary:")
    print(f"  Vulnerable households (Struggling + Severely Struggling): {vulnerable_count:,} ({vulnerable_percentage:.1f}%)")
    print(f"  Non-vulnerable households (On Track + At Risk): {len(df) - vulnerable_count:,} ({100 - vulnerable_percentage:.1f}%)")
    
    return df

def analyze_target_distribution(df):
    """Analyze the target variable distribution"""
    
    print("\n5. TARGET VARIABLE DETAILED ANALYSIS...")
    
    target_col = 'HHIncome+Consumption+Residues/Day'
    
    # Statistical analysis by status
    print("\nStatistical analysis by ProgressStatus:")
    for status in df['ProgressStatus'].unique():
        if status != 'Unknown':
            subset = df[df['ProgressStatus'] == status][target_col]
            print(f"\n{status}:")
            print(f"  Count: {len(subset):,}")
            print(f"  Mean: ${subset.mean():.2f}/day")
            print(f"  Median: ${subset.median():.2f}/day")
            print(f"  Min: ${subset.min():.2f}/day")
            print(f"  Max: ${subset.max():.2f}/day")
    
    # Geographic distribution analysis
    print("\n6. GEOGRAPHIC VULNERABILITY ANALYSIS...")
    
    # By District
    print("\nVulnerability by District:")
    district_vulnerability = df.groupby('District').agg({
        'ProgressStatus': lambda x: (x.isin(['Struggling', 'Severely Struggling'])).sum(),
        'District': 'count'
    }).rename(columns={'District': 'Total_Households'})
    district_vulnerability['Vulnerability_Rate'] = (
        district_vulnerability['ProgressStatus'] / district_vulnerability['Total_Households'] * 100
    )
    
    for district in district_vulnerability.index:
        total = district_vulnerability.loc[district, 'Total_Households']
        vulnerable = district_vulnerability.loc[district, 'ProgressStatus']
        rate = district_vulnerability.loc[district, 'Vulnerability_Rate']
        print(f"  {district}: {vulnerable}/{total} households ({rate:.1f}% vulnerable)")
    
    # Household size analysis
    print("\n7. HOUSEHOLD SIZE AND VULNERABILITY ANALYSIS...")
    
    vulnerability_by_size = df.groupby('ProgressStatus')['HouseholdSize'].agg(['mean', 'median', 'std']).round(2)
    print("\nHousehold size by vulnerability status:")
    for status in vulnerability_by_size.index:
        if status != 'Unknown':
            mean_size = vulnerability_by_size.loc[status, 'mean']
            median_size = vulnerability_by_size.loc[status, 'median']
            print(f"  {status}: Mean {mean_size} members, Median {median_size} members")
    
    return df

def create_visualizations(df):
    """Create visualizations for the target variable"""
    
    print("\n8. CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Household Vulnerability Analysis - Target Variable Distribution', fontsize=16, fontweight='bold')
    
    # 1. ProgressStatus Distribution (Pie Chart)
    status_counts = df['ProgressStatus'].value_counts()
    colors = ['#2E8B57', '#FFD700', '#FF6347', '#DC143C']  # Green to Red
    
    axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Distribution of Progress Status')
    
    # 2. Daily Income Distribution by Status
    target_col = 'HHIncome+Consumption+Residues/Day'
    for i, status in enumerate(['On Track', 'At Risk', 'Struggling', 'Severely Struggling']):
        if status in df['ProgressStatus'].values:
            data = df[df['ProgressStatus'] == status][target_col]
            axes[0, 1].hist(data, alpha=0.6, label=status, bins=20, color=colors[i])
    
    axes[0, 1].set_xlabel('Daily Income (USD)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Daily Income Distribution by Progress Status')
    axes[0, 1].legend()
    axes[0, 1].axvline(x=2.15, color='green', linestyle='--', alpha=0.7, label='$2.15 threshold')
    axes[0, 1].axvline(x=1.77, color='orange', linestyle='--', alpha=0.7, label='$1.77 threshold')
    axes[0, 1].axvline(x=1.25, color='red', linestyle='--', alpha=0.7, label='$1.25 threshold')
    
    # 3. Vulnerability by District
    district_vuln = df.groupby('District').apply(
        lambda x: (x['ProgressStatus'].isin(['Struggling', 'Severely Struggling'])).sum() / len(x) * 100
    ).sort_values(ascending=True)
    
    axes[1, 0].barh(range(len(district_vuln)), district_vuln.values, color='coral')
    axes[1, 0].set_yticks(range(len(district_vuln)))
    axes[1, 0].set_yticklabels(district_vuln.index)
    axes[1, 0].set_xlabel('Vulnerability Rate (%)')
    axes[1, 0].set_title('Vulnerability Rate by District')
    
    # 4. Household Size vs Vulnerability
    df_boxplot = df[df['ProgressStatus'] != 'Unknown']
    sns.boxplot(data=df_boxplot, x='ProgressStatus', y='HouseholdSize', ax=axes[1, 1])
    axes[1, 1].set_title('Household Size by Progress Status')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('progress_status_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to 'progress_status_analysis.png'")
    plt.close()

def save_results(df):
    """Save the dataset with ProgressStatus and create summary"""
    
    print("\n9. SAVING RESULTS...")
    
    # Save dataset with ProgressStatus
    output_file = 'data_with_proper_progress_status.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Saved dataset with ProgressStatus to '{output_file}'")
    
    # Create and save progress status summary
    summary_data = []
    
    for status in ['On Track', 'At Risk', 'Struggling', 'Severely Struggling']:
        if status in df['ProgressStatus'].values:
            subset = df[df['ProgressStatus'] == status]
            target_col = 'HHIncome+Consumption+Residues/Day'
            
            summary_data.append({
                'ProgressStatus': status,
                'Count': len(subset),
                'Percentage': len(subset) / len(df) * 100,
                'Mean_Daily_Income': subset[target_col].mean(),
                'Median_Daily_Income': subset[target_col].median(),
                'Mean_Household_Size': subset['HouseholdSize'].mean(),
                'Vulnerability_Level': 'High' if status in ['Struggling', 'Severely Struggling'] else 'Low'
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('progress_status_summary.csv', index=False)
    print("✓ Saved progress status summary to 'progress_status_summary.csv'")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TARGET VARIABLE CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    total_households = len(df)
    vulnerable_count = len(df[df['ProgressStatus'].isin(['Struggling', 'Severely Struggling'])])
    vulnerability_rate = (vulnerable_count / total_households) * 100
    
    print(f"\nFinal Results:")
    print(f"• Total households analyzed: {total_households:,}")
    print(f"• Households with ProgressStatus: {len(df[df['ProgressStatus'] != 'Unknown']):,}")
    print(f"• Overall vulnerability rate: {vulnerability_rate:.1f}%")
    print(f"• Average daily income: ${df['HHIncome+Consumption+Residues/Day'].mean():.2f}")
    
    print(f"\nDistribution breakdown:")
    for status in ['On Track', 'At Risk', 'Struggling', 'Severely Struggling']:
        if status in df['ProgressStatus'].values:
            count = len(df[df['ProgressStatus'] == status])
            percentage = count / total_households * 100
            print(f"• {status}: {count:,} households ({percentage:.1f}%)")
    
    print(f"\nFiles generated:")
    print(f"• {output_file} - Dataset with ProgressStatus variable")
    print(f"• progress_status_summary.csv - Statistical summary")
    print(f"• progress_status_analysis.png - Visualization")
    
    print(f"\nNext steps:")
    print(f"1. Run '03_comprehensive_eda.py' for detailed exploratory analysis")
    print(f"2. Run '04_ml_modeling.py' for machine learning model development")

def main():
    """Main execution function"""
    print("Starting target variable creation...")
    
    # Load data
    df, dict_df = load_data()
    
    if df is not None and dict_df is not None:
        # Create ProgressStatus variable
        df = create_progress_status(df)
        
        if df is not None:
            # Analyze target distribution
            df = analyze_target_distribution(df)
            
            # Create visualizations
            create_visualizations(df)
            
            # Save results
            save_results(df)
        else:
            print("✗ Failed to create ProgressStatus variable")
    else:
        print("✗ Failed to load data files")

if __name__ == "__main__":
    main() 