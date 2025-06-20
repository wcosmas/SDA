#!/usr/bin/env python3
"""
RTV Senior Data Scientist Technical Assessment - Part A
03. Comprehensive Exploratory Data Analysis

This script performs detailed EDA on the household vulnerability dataset
including feature analysis, correlation studies, and vulnerability patterns.

Author: Data Science Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the dataset with ProgressStatus variable"""
    
    print("=" * 80)
    print("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - COMPREHENSIVE EDA")
    print("=" * 80)
    
    try:
        print("\n1. LOADING PROCESSED DATASET...")
        df = pd.read_csv('data_with_proper_progress_status.csv')
        print(f"✓ Successfully loaded dataset: {df.shape[0]} households, {df.shape[1]} variables")
        
        # Verify ProgressStatus exists
        if 'ProgressStatus' not in df.columns:
            print("✗ ProgressStatus variable not found. Please run 02_target_variable_creation.py first.")
            return None
            
        print(f"✓ ProgressStatus distribution:")
        for status, count in df['ProgressStatus'].value_counts().items():
            percentage = count / len(df) * 100
            print(f"  {status}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except FileNotFoundError:
        print("✗ Processed dataset not found. Please run 02_target_variable_creation.py first.")
        return None

def analyze_feature_categories(df):
    """Categorize and analyze different types of features"""
    
    print("\n2. FEATURE CATEGORIZATION AND ANALYSIS...")
    
    # Categorize features based on content and data dictionary knowledge
    feature_categories = {
        'Geographic': ['District', 'Cluster', 'Village'],
        'Household_Demographics': ['HouseholdSize', 'hhh_sex', 'hhh_read_write'],
        'Infrastructure_Access': ['TimeToOPD', 'TimeToWater'],
        'Agricultural_Assets': ['AgricultureLand', 'Season1CropsPlanted', 'Season2CropsPlanted', 
                               'PerennialCropsGrown', 'perennial_cropping', 'household_fertilizer'],
        'Income_Sources': ['Season1VegetableIncome', 'Season2VegatableIncome', 'VegetableIncome',
                          'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 'CasualLabour',
                          'RemittancesAndGifts', 'RentIncome', 'SeasonalCropIncome', 
                          'PerenialCropIncome', 'LivestockIncome'],
        'Agricultural_Value': ['Season1VegetableValue', 'Season2VegetableValue', 'SeasonalVegetableValue',
                              'Season1AgricValue', 'Season2AgricValue', 'SeasonalAgricValue',
                              'PerennialAgricValue', 'AgricValue', 'LivestockIncomeConsumed', 
                              'LivestockAssetValue'],
        'Total_Economic': ['HouseholdIcome', 'Consumption+Residues', 'HHIncome+Consumption+Residues',
                          'HHIncome+Consumption+Assets+Residues', 'Assets', 'Assets.1'],
        'Daily_Economic': ['HHIncome/Day', 'Consumption/Day', 'HHIncome+Consumption+Residues/Day',
                          'HHIncome+Consumption+Assets+Residues/Day'],
        'Technology_Assets': ['radios_owned', 'phones_owned'],
        'Employment': ['business_number', 'work_casual', 'work_salaried'],
        'Financial_Services': ['VSLA_Profits', 'VSLA_Profits.1', 'save_mode_7', 'Loan_from'],
        'Housing_Infrastructure': ['Material_walls', 'latrine_constructed', 'tippy_tap_available',
                                 'soap_ash_available', 'standard_hangline', 'kitchen_house',
                                 'bathroom_constructed', 'swept_compound', 'dish_rack_present'],
        'Agricultural_Practices': ['daily_meals', 'composts', 'non_bio_waste_mgt_present',
                                 'apply_liquid_manure', 'organic_pesticide_expenditure',
                                 'water_control_practise', 'soil_management'],
        'Food_Security': ['food_banana_wilt_diseases', 'postharvest_food_storage']
    }
    
    print("Feature categories identified:")
    total_categorized = 0
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        print(f"  {category}: {len(available_features)} features")
        total_categorized += len(available_features)
    
    print(f"\nTotal categorized features: {total_categorized}")
    print(f"Uncategorized features: {len(df.columns) - total_categorized - 1}")  # -1 for ProgressStatus
    
    return feature_categories

def analyze_vulnerability_by_categories(df, feature_categories):
    """Analyze vulnerability patterns across different feature categories"""
    
    print("\n3. VULNERABILITY ANALYSIS BY FEATURE CATEGORIES...")
    
    vulnerable_statuses = ['Struggling', 'Severely Struggling']
    df['is_vulnerable'] = df['ProgressStatus'].isin(vulnerable_statuses).astype(int)
    
    category_insights = {}
    
    for category, features in feature_categories.items():
        print(f"\n{category.upper().replace('_', ' ')} ANALYSIS:")
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            print("  No features available in this category")
            continue
            
        category_insights[category] = {}
        
        for feature in available_features[:5]:  # Analyze top 5 features per category
            try:
                if df[feature].dtype in ['object', 'category']:
                    # Categorical analysis
                    if df[feature].nunique() <= 10:  # Only for reasonable number of categories
                        crosstab = pd.crosstab(df[feature], df['is_vulnerable'], normalize='index')
                        if 1 in crosstab.columns:
                            vuln_rates = crosstab[1].sort_values(ascending=False)
                            print(f"  {feature} - Vulnerability rates by category:")
                            for cat, rate in vuln_rates.head(3).items():
                                print(f"    {cat}: {rate:.1%}")
                                
                elif pd.api.types.is_numeric_dtype(df[feature]):
                    # Numerical analysis
                    vulnerable_mean = df[df['is_vulnerable'] == 1][feature].mean()
                    non_vulnerable_mean = df[df['is_vulnerable'] == 0][feature].mean()
                    
                    if not pd.isna(vulnerable_mean) and not pd.isna(non_vulnerable_mean):
                        diff = vulnerable_mean - non_vulnerable_mean
                        print(f"  {feature}:")
                        print(f"    Vulnerable: {vulnerable_mean:.2f}, Non-vulnerable: {non_vulnerable_mean:.2f}")
                        print(f"    Difference: {diff:.2f}")
                        
                        category_insights[category][feature] = {
                            'vulnerable_mean': vulnerable_mean,
                            'non_vulnerable_mean': non_vulnerable_mean,
                            'difference': diff
                        }
                        
            except Exception as e:
                print(f"    Error analyzing {feature}: {str(e)}")
                continue
    
    return category_insights, df

def correlation_analysis(df):
    """Perform correlation analysis between features and vulnerability"""
    
    print("\n4. CORRELATION ANALYSIS...")
    
    # Select numerical features for correlation analysis
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and target variable
    exclude_cols = ['HouseHoldID', 'ProgressStatus', 'is_vulnerable']
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    
    print(f"Analyzing correlations for {len(numerical_features)} numerical features...")
    
    # Calculate correlations with vulnerability
    correlations = []
    target_col = 'HHIncome+Consumption+Residues/Day'
    
    for feature in numerical_features:
        try:
            # Correlation with daily income (continuous target)
            corr_income = df[feature].corr(df[target_col])
            
            # Correlation with vulnerability (binary)
            corr_vuln = df[feature].corr(df['is_vulnerable'])
            
            correlations.append({
                'Feature': feature,
                'Correlation_with_Income': corr_income,
                'Correlation_with_Vulnerability': corr_vuln,
                'Abs_Correlation_Income': abs(corr_income) if not pd.isna(corr_income) else 0,
                'Abs_Correlation_Vulnerability': abs(corr_vuln) if not pd.isna(corr_vuln) else 0
            })
            
        except Exception as e:
            print(f"  Error calculating correlation for {feature}: {str(e)}")
            continue
    
    corr_df = pd.DataFrame(correlations)
    
    # Sort by absolute correlation with vulnerability
    corr_df = corr_df.sort_values('Abs_Correlation_Vulnerability', ascending=False)
    
    print("\nTop 15 features by correlation with vulnerability:")
    print("Feature                                    | Income Corr | Vuln Corr")
    print("-" * 70)
    for _, row in corr_df.head(15).iterrows():
        feature_name = row['Feature'][:35].ljust(35)
        income_corr = f"{row['Correlation_with_Income']:.3f}".rjust(9)
        vuln_corr = f"{row['Correlation_with_Vulnerability']:.3f}".rjust(9)
        print(f"{feature_name} | {income_corr} | {vuln_corr}")
    
    # Save correlation analysis
    corr_df.to_csv('feature_correlation_analysis.csv', index=False)
    print("\n✓ Saved correlation analysis to 'feature_correlation_analysis.csv'")
    
    return corr_df

def geographic_vulnerability_analysis(df):
    """Detailed geographic analysis of vulnerability patterns"""
    
    print("\n5. GEOGRAPHIC VULNERABILITY ANALYSIS...")
    
    # District-level analysis
    print("\nDistrict-level vulnerability analysis:")
    district_analysis = df.groupby('District').agg({
        'is_vulnerable': ['count', 'sum', 'mean'],
        'HHIncome+Consumption+Residues/Day': ['mean', 'median'],
        'HouseholdSize': 'mean',
        'AgricultureLand': 'mean'
    }).round(2)
    
    district_analysis.columns = [
        'Total_HH', 'Vulnerable_HH', 'Vulnerability_Rate',
        'Avg_Daily_Income', 'Median_Daily_Income', 'Avg_HH_Size', 'Avg_Land_Size'
    ]
    
    print("\nDistrict Summary:")
    print("District    | Total | Vuln | Rate  | Avg Income | HH Size | Land")
    print("-" * 70)
    for district, row in district_analysis.iterrows():
        district_name = district[:10].ljust(10)
        total = f"{int(row['Total_HH'])}".rjust(5)
        vuln = f"{int(row['Vulnerable_HH'])}".rjust(4)
        rate = f"{row['Vulnerability_Rate']:.1%}".rjust(5)
        income = f"${row['Avg_Daily_Income']:.2f}".rjust(8)
        hh_size = f"{row['Avg_HH_Size']:.1f}".rjust(5)
        land = f"{row['Avg_Land_Size']:.1f}".rjust(4)
        print(f"{district_name} | {total} | {vuln} | {rate} | {income} | {hh_size} | {land}")
    
    # Save district analysis
    district_analysis.to_csv('district_vulnerability_analysis.csv')
    print("\n✓ Saved district analysis to 'district_vulnerability_analysis.csv'")
    
    return district_analysis

def infrastructure_analysis(df):
    """Analyze infrastructure and service access patterns"""
    
    print("\n6. INFRASTRUCTURE AND SERVICE ACCESS ANALYSIS...")
    
    infrastructure_features = [
        'TimeToOPD', 'TimeToWater', 'latrine_constructed', 'tippy_tap_available',
        'soap_ash_available', 'bathroom_constructed', 'kitchen_house'
    ]
    
    print("\nInfrastructure access by vulnerability status:")
    
    for feature in infrastructure_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            
            if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 2:
                # Continuous variable (like time)
                vuln_mean = df[df['is_vulnerable'] == 1][feature].mean()
                non_vuln_mean = df[df['is_vulnerable'] == 0][feature].mean()
                print(f"  Vulnerable households: {vuln_mean:.1f}")
                print(f"  Non-vulnerable households: {non_vuln_mean:.1f}")
                print(f"  Difference: {vuln_mean - non_vuln_mean:.1f}")
                
            else:
                # Binary variable
                access_rate_vuln = df[df['is_vulnerable'] == 1][feature].mean()
                access_rate_non_vuln = df[df['is_vulnerable'] == 0][feature].mean()
                print(f"  Vulnerable households access rate: {access_rate_vuln:.1%}")
                print(f"  Non-vulnerable households access rate: {access_rate_non_vuln:.1%}")
                print(f"  Gap: {access_rate_non_vuln - access_rate_vuln:.1%}")

def agricultural_analysis(df):
    """Analyze agricultural patterns and vulnerability"""
    
    print("\n7. AGRICULTURAL ANALYSIS...")
    
    agricultural_features = [
        'AgricultureLand', 'Season1CropsPlanted', 'Season2CropsPlanted',
        'SeasonalCropIncome', 'PerenialCropIncome', 'AgricValue',
        'perennial_cropping', 'household_fertilizer'
    ]
    
    print("\nAgricultural patterns by vulnerability status:")
    
    for feature in agricultural_features:
        if feature in df.columns:
            print(f"\n{feature}:")
            
            if df[feature].dtype in ['int64', 'float64']:
                vuln_mean = df[df['is_vulnerable'] == 1][feature].mean()
                non_vuln_mean = df[df['is_vulnerable'] == 0][feature].mean()
                
                if not pd.isna(vuln_mean) and not pd.isna(non_vuln_mean):
                    print(f"  Vulnerable households: {vuln_mean:.2f}")
                    print(f"  Non-vulnerable households: {non_vuln_mean:.2f}")
                    print(f"  Difference: {vuln_mean - non_vuln_mean:.2f}")
                    
                    # Calculate percentage difference
                    if non_vuln_mean != 0:
                        pct_diff = ((vuln_mean - non_vuln_mean) / non_vuln_mean) * 100
                        print(f"  Percentage difference: {pct_diff:.1f}%")

def create_comprehensive_visualizations(df, corr_df):
    """Create comprehensive visualizations for EDA"""
    
    print("\n8. CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Correlation heatmap (top features)
    plt.subplot(3, 3, 1)
    top_features = corr_df.head(10)['Feature'].tolist()
    top_features.append('HHIncome+Consumption+Residues/Day')
    top_features.append('is_vulnerable')
    
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Top Features)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. Income distribution by district
    plt.subplot(3, 3, 2)
    df.boxplot(column='HHIncome+Consumption+Residues/Day', by='District', ax=plt.gca())
    plt.title('Daily Income Distribution by District')
    plt.suptitle('')  # Remove default title
    plt.xticks(rotation=45)
    plt.ylabel('Daily Income (USD)')
    
    # 3. Vulnerability rate by household size
    plt.subplot(3, 3, 3)
    hh_size_vuln = df.groupby('HouseholdSize')['is_vulnerable'].agg(['count', 'mean']).reset_index()
    hh_size_vuln = hh_size_vuln[hh_size_vuln['count'] >= 10]  # Only sizes with 10+ households
    
    plt.scatter(hh_size_vuln['HouseholdSize'], hh_size_vuln['mean'], 
               s=hh_size_vuln['count']*2, alpha=0.6)
    plt.xlabel('Household Size')
    plt.ylabel('Vulnerability Rate')
    plt.title('Vulnerability Rate by Household Size')
    
    # 4. Infrastructure access comparison
    plt.subplot(3, 3, 4)
    infrastructure_cols = ['latrine_constructed', 'tippy_tap_available', 'soap_ash_available', 
                          'bathroom_constructed', 'kitchen_house']
    
    vuln_rates = []
    non_vuln_rates = []
    labels = []
    
    for col in infrastructure_cols:
        if col in df.columns:
            vuln_rate = df[df['is_vulnerable'] == 1][col].mean()
            non_vuln_rate = df[df['is_vulnerable'] == 0][col].mean()
            vuln_rates.append(vuln_rate)
            non_vuln_rates.append(non_vuln_rate)
            labels.append(col.replace('_', ' ').title())
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, vuln_rates, width, label='Vulnerable', alpha=0.8)
    plt.bar(x + width/2, non_vuln_rates, width, label='Non-vulnerable', alpha=0.8)
    plt.xlabel('Infrastructure Type')
    plt.ylabel('Access Rate')
    plt.title('Infrastructure Access by Vulnerability Status')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    
    # 5. Agricultural income comparison
    plt.subplot(3, 3, 5)
    ag_income_cols = ['SeasonalCropIncome', 'PerenialCropIncome', 'LivestockIncome', 'VegetableIncome']
    
    vuln_ag_income = []
    non_vuln_ag_income = []
    ag_labels = []
    
    for col in ag_income_cols:
        if col in df.columns:
            vuln_mean = df[df['is_vulnerable'] == 1][col].mean()
            non_vuln_mean = df[df['is_vulnerable'] == 0][col].mean()
            vuln_ag_income.append(vuln_mean)
            non_vuln_ag_income.append(non_vuln_mean)
            ag_labels.append(col.replace('Income', '').replace('Seasonal', 'Season.'))
    
    x = np.arange(len(ag_labels))
    plt.bar(x - width/2, vuln_ag_income, width, label='Vulnerable', alpha=0.8)
    plt.bar(x + width/2, non_vuln_ag_income, width, label='Non-vulnerable', alpha=0.8)
    plt.xlabel('Income Source')
    plt.ylabel('Average Annual Income (USD)')
    plt.title('Agricultural Income Sources by Vulnerability')
    plt.xticks(x, ag_labels, rotation=45, ha='right')
    plt.legend()
    
    # 6. Time to services comparison
    plt.subplot(3, 3, 6)
    time_cols = ['TimeToOPD', 'TimeToWater']
    vuln_times = [df[df['is_vulnerable'] == 1][col].mean() for col in time_cols if col in df.columns]
    non_vuln_times = [df[df['is_vulnerable'] == 0][col].mean() for col in time_cols if col in df.columns]
    time_labels = [col.replace('TimeTo', '') for col in time_cols if col in df.columns]
    
    x = np.arange(len(time_labels))
    plt.bar(x - width/2, vuln_times, width, label='Vulnerable', alpha=0.8)
    plt.bar(x + width/2, non_vuln_times, width, label='Non-vulnerable', alpha=0.8)
    plt.xlabel('Service Type')
    plt.ylabel('Average Time (minutes)')
    plt.title('Time to Services by Vulnerability Status')
    plt.xticks(x, time_labels)
    plt.legend()
    
    # 7. Asset ownership comparison
    plt.subplot(3, 3, 7)
    asset_cols = ['radios_owned', 'phones_owned', 'work_casual', 'work_salaried']
    
    vuln_asset_rates = []
    non_vuln_asset_rates = []
    asset_labels = []
    
    for col in asset_cols:
        if col in df.columns:
            vuln_rate = df[df['is_vulnerable'] == 1][col].mean()
            non_vuln_rate = df[df['is_vulnerable'] == 0][col].mean()
            vuln_asset_rates.append(vuln_rate)
            non_vuln_asset_rates.append(non_vuln_rate)
            asset_labels.append(col.replace('_', ' ').title())
    
    x = np.arange(len(asset_labels))
    plt.bar(x - width/2, vuln_asset_rates, width, label='Vulnerable', alpha=0.8)
    plt.bar(x + width/2, non_vuln_asset_rates, width, label='Non-vulnerable', alpha=0.8)
    plt.xlabel('Asset/Employment Type')
    plt.ylabel('Ownership/Employment Rate')
    plt.title('Asset Ownership & Employment by Vulnerability')
    plt.xticks(x, asset_labels, rotation=45, ha='right')
    plt.legend()
    
    # 8. Feature importance (correlation-based)
    plt.subplot(3, 3, 8)
    top_10_features = corr_df.head(10)
    
    colors = ['red' if x < 0 else 'green' for x in top_10_features['Correlation_with_Vulnerability']]
    plt.barh(range(len(top_10_features)), top_10_features['Correlation_with_Vulnerability'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_10_features)), [f[:20] for f in top_10_features['Feature']])
    plt.xlabel('Correlation with Vulnerability')
    plt.title('Top 10 Features - Correlation with Vulnerability')
    plt.grid(axis='x', alpha=0.3)
    
    # 9. District vulnerability summary
    plt.subplot(3, 3, 9)
    district_vuln = df.groupby('District')['is_vulnerable'].agg(['count', 'mean']).reset_index()
    
    colors = plt.cm.Reds(district_vuln['mean'])
    bars = plt.bar(district_vuln['District'], district_vuln['mean'], color=colors, alpha=0.8)
    plt.xlabel('District')
    plt.ylabel('Vulnerability Rate')
    plt.title('Vulnerability Rate by District')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, district_vuln['count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comprehensive_eda_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comprehensive visualization to 'comprehensive_eda_analysis.png'")
    plt.close()

def save_eda_summary(df, feature_categories, corr_df, district_analysis):
    """Save comprehensive EDA summary"""
    
    print("\n9. SAVING EDA SUMMARY...")
    
    # Create comprehensive summary
    eda_summary = {
        'dataset_overview': {
            'total_households': len(df),
            'total_features': len(df.columns) - 1,  # Exclude ProgressStatus
            'vulnerability_rate': df['is_vulnerable'].mean(),
            'data_completeness': ((df.notna().sum().sum()) / (len(df) * len(df.columns))) * 100
        },
        'vulnerability_distribution': df['ProgressStatus'].value_counts().to_dict(),
        'top_vulnerability_predictors': corr_df.head(10)[['Feature', 'Correlation_with_Vulnerability']].to_dict('records'),
        'district_patterns': district_analysis.to_dict('index'),
        'key_insights': [
            f"Overall vulnerability rate: {df['is_vulnerable'].mean():.1%}",
            f"Highest risk district: {district_analysis['Vulnerability_Rate'].idxmax()} ({district_analysis['Vulnerability_Rate'].max():.1%})",
            f"Lowest risk district: {district_analysis['Vulnerability_Rate'].idxmin()} ({district_analysis['Vulnerability_Rate'].min():.1%})",
            f"Average household size: {df['HouseholdSize'].mean():.1f} members",
            f"Average daily income: ${df['HHIncome+Consumption+Residues/Day'].mean():.2f}",
            f"Income gap: ${df[df['is_vulnerable']==0]['HHIncome+Consumption+Residues/Day'].mean() - df[df['is_vulnerable']==1]['HHIncome+Consumption+Residues/Day'].mean():.2f}/day"
        ]
    }
    
    # Save as text summary
    with open('comprehensive_eda_summary.txt', 'w') as f:
        f.write("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - EDA SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        for key, value in eda_summary['dataset_overview'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
        
        f.write("\nVULNERABILITY DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for status, count in eda_summary['vulnerability_distribution'].items():
            percentage = (count / eda_summary['dataset_overview']['total_households']) * 100
            f.write(f"{status}: {count:,} ({percentage:.1f}%)\n")
        
        f.write("\nTOP VULNERABILITY PREDICTORS:\n")
        f.write("-" * 35 + "\n")
        for predictor in eda_summary['top_vulnerability_predictors']:
            f.write(f"{predictor['Feature']}: {predictor['Correlation_with_Vulnerability']:.3f}\n")
        
        f.write("\nKEY INSIGHTS:\n")
        f.write("-" * 15 + "\n")
        for insight in eda_summary['key_insights']:
            f.write(f"• {insight}\n")
    
    print("✓ Saved EDA summary to 'comprehensive_eda_summary.txt'")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EDA COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\nDataset Summary:")
    print(f"• {len(df):,} households analyzed across {df['District'].nunique()} districts")
    print(f"• {df['is_vulnerable'].sum():,} vulnerable households ({df['is_vulnerable'].mean():.1%})")
    print(f"• {len(df.columns)-1} features analyzed across {len(feature_categories)} categories")
    
    print(f"\nKey Findings:")
    print(f"• Highest vulnerability: {district_analysis['Vulnerability_Rate'].idxmax()} ({district_analysis['Vulnerability_Rate'].max():.1%})")
    print(f"• Lowest vulnerability: {district_analysis['Vulnerability_Rate'].idxmin()} ({district_analysis['Vulnerability_Rate'].min():.1%})")
    print(f"• Top predictor: {corr_df.iloc[0]['Feature']} (correlation: {corr_df.iloc[0]['Correlation_with_Vulnerability']:.3f})")
    
    income_gap = df[df['is_vulnerable']==0]['HHIncome+Consumption+Residues/Day'].mean() - df[df['is_vulnerable']==1]['HHIncome+Consumption+Residues/Day'].mean()
    print(f"• Income gap: ${income_gap:.2f}/day between vulnerable and non-vulnerable")
    
    print(f"\nFiles Generated:")
    print(f"• feature_correlation_analysis.csv - Detailed correlation analysis")
    print(f"• district_vulnerability_analysis.csv - Geographic analysis")
    print(f"• comprehensive_eda_analysis.png - Visual analysis")
    print(f"• comprehensive_eda_summary.txt - Executive summary")
    
    print(f"\nNext Steps:")
    print(f"1. Run '04_ml_modeling.py' for machine learning model development")
    print(f"2. Use insights to guide feature selection and engineering")
    print(f"3. Consider geographic and categorical factors in modeling")

def main():
    """Main execution function"""
    print("Starting comprehensive EDA...")
    
    # Load processed data
    df = load_processed_data()
    
    if df is not None:
        # Analyze feature categories
        feature_categories = analyze_feature_categories(df)
        
        # Vulnerability analysis by categories
        category_insights, df = analyze_vulnerability_by_categories(df, feature_categories)
        
        # Correlation analysis
        corr_df = correlation_analysis(df)
        
        # Geographic analysis
        district_analysis = geographic_vulnerability_analysis(df)
        
        # Infrastructure analysis
        infrastructure_analysis(df)
        
        # Agricultural analysis
        agricultural_analysis(df)
        
        # Create visualizations
        create_comprehensive_visualizations(df, corr_df)
        
        # Save summary
        save_eda_summary(df, feature_categories, corr_df, district_analysis)
        
    else:
        print("✗ Failed to load processed data. Please run previous scripts first.")

if __name__ == "__main__":
    main() 