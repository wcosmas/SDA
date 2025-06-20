"""
RTV Senior Data Scientist Technical Assessment - Part A
05. Model Evaluation and Business Insights

This script provides comprehensive evaluation of the trained model and generates
actionable business insights for the RTV program.

Author: Data Science Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load the trained model and test data"""
    
    print("=" * 80)
    print("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - MODEL EVALUATION")
    print("=" * 80)
    
    try:
        print("\n1. LOADING MODEL AND DATA...")
        
        # Load the trained model
        model = joblib.load('best_vulnerability_model.pkl')
        print("✓ Successfully loaded trained model")
        
        # Load the dataset
        df = pd.read_csv('data_with_proper_progress_status.csv')
        print(f"✓ Successfully loaded dataset: {df.shape[0]} households")
        
        # Load model information
        model_info = pd.read_csv('model_features_info.csv')
        model_type = model_info[model_info['Metric'] == 'Model Type']['Value'].iloc[0]
        print(f"✓ Model type: {model_type}")
        
        # Load feature importance
        feature_importance = pd.read_csv('feature_importance_analysis.csv')
        print(f"✓ Loaded feature importance for {len(feature_importance)} features")
        
        return model, df, model_info, feature_importance
        
    except FileNotFoundError as e:
        print(f"✗ Error loading files: {e}")
        print("Please ensure you have run all previous scripts.")
        return None, None, None, None

def generate_predictions_for_all_households(model, df):
    """Generate predictions for all households in the dataset"""
    
    print("\n2. GENERATING PREDICTIONS FOR ALL HOUSEHOLDS...")
    
    # Create the same features used in training
    # Numeric features
    numeric_features = [
        'HouseholdSize', 'TimeToOPD', 'TimeToWater', 'AgricultureLand',
        'Season1CropsPlanted', 'Season2CropsPlanted', 'PerennialCropsGrown',
        'Season1VegetableIncome', 'Season2VegatableIncome', 'VegetableIncome',
        'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 'CasualLabour',
        'RemittancesAndGifts', 'RentIncome', 'SeasonalCropIncome',
        'PerenialCropIncome', 'LivestockIncome', 'AgricValue',
        'HouseholdIcome', 'Assets.1'
    ]
    
    categorical_features = [
        'District', 'hhh_sex', 'hhh_read_write', 'Material_walls'
    ]
    
    binary_features = [
        'radios_owned', 'phones_owned', 'work_casual', 'work_salaried',
        'latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
        'standard_hangline', 'kitchen_house', 'bathroom_constructed',
        'swept_compound', 'dish_rack_present', 'perennial_cropping',
        'household_fertilizer', 'non_bio_waste_mgt_present',
        'apply_liquid_manure', 'water_control_practise', 'soil_management',
        'postharvest_food_storage', 'save_mode_7'
    ]
    
    # Filter features that exist in the dataset
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    
    # Create engineered features (same as in training)
    if 'HouseholdSize' in df.columns and 'HouseholdIcome' in df.columns:
        df['income_per_capita'] = df['HouseholdIcome'] / df['HouseholdSize'].replace(0, 1)
        numeric_features.append('income_per_capita')
    
    if 'AgricultureLand' in df.columns and 'AgricValue' in df.columns:
        df['agric_productivity'] = df['AgricValue'] / (df['AgricultureLand'].replace(0, 0.1))
        numeric_features.append('agric_productivity')
    
    if 'HouseholdSize' in df.columns:
        df['household_size_category'] = pd.cut(df['HouseholdSize'], 
                                             bins=[0, 3, 5, 7, float('inf')], 
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])
        categorical_features.append('household_size_category')
    
    # Asset ownership score
    asset_cols = ['radios_owned', 'phones_owned']
    available_asset_cols = [col for col in asset_cols if col in df.columns]
    if available_asset_cols:
        df['asset_ownership_score'] = df[available_asset_cols].sum(axis=1)
        numeric_features.append('asset_ownership_score')
    
    # Infrastructure score
    infra_cols = ['latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
                  'bathroom_constructed', 'kitchen_house']
    available_infra_cols = [col for col in infra_cols if col in df.columns]
    if available_infra_cols:
        df['infrastructure_score'] = df[available_infra_cols].sum(axis=1)
        numeric_features.append('infrastructure_score')
    
    # Prepare features for prediction
    all_features = numeric_features + categorical_features + binary_features
    X = df[all_features]
    
    # Generate predictions
    predictions = model.predict(X)
    prediction_probabilities = model.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    df['predicted_vulnerable'] = predictions
    df['vulnerability_probability'] = prediction_probabilities
    
    # Create risk categories based on probability
    df['risk_category'] = pd.cut(df['vulnerability_probability'],
                                bins=[0, 0.3, 0.6, 0.8, 1.0],
                                labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    
    print(f"✓ Generated predictions for {len(df)} households")
    print(f"✓ Predicted vulnerable households: {predictions.sum():,} ({predictions.mean():.1%})")
    
    # Prediction accuracy (on households with known status)
    df['actual_vulnerable'] = df['ProgressStatus'].isin(['Struggling', 'Severely Struggling']).astype(int)
    accuracy = (df['predicted_vulnerable'] == df['actual_vulnerable']).mean()
    print(f"✓ Overall prediction accuracy: {accuracy:.1%}")
    
    return df

def analyze_prediction_accuracy(df):
    """Analyze prediction accuracy and model performance"""
    
    print("\n3. MODEL PERFORMANCE ANALYSIS...")
    
    # Classification report
    y_true = df['actual_vulnerable']
    y_pred = df['predicted_vulnerable']
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-vulnerable', 'Vulnerable']))
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"  True Negatives (Correctly identified non-vulnerable): {cm[0,0]:,}")
    print(f"  False Positives (Incorrectly predicted vulnerable): {cm[0,1]:,}")
    print(f"  False Negatives (Missed vulnerable households): {cm[1,0]:,}")
    print(f"  True Positives (Correctly identified vulnerable): {cm[1,1]:,}")
    
    # Calculate business metrics
    total_vulnerable = y_true.sum()
    total_predicted_vulnerable = y_pred.sum()
    correctly_identified_vulnerable = cm[1,1]
    
    coverage_rate = correctly_identified_vulnerable / total_vulnerable
    precision_rate = correctly_identified_vulnerable / total_predicted_vulnerable if total_predicted_vulnerable > 0 else 0
    
    print(f"\nBusiness Impact Metrics:")
    print(f"  Vulnerable household coverage: {coverage_rate:.1%} ({correctly_identified_vulnerable:,}/{total_vulnerable:,})")
    print(f"  Program targeting precision: {precision_rate:.1%} ({correctly_identified_vulnerable:,}/{total_predicted_vulnerable:,})")
    print(f"  Households requiring intervention: {total_predicted_vulnerable:,}")

def geographic_insights(df):
    """Generate geographic insights for program targeting"""
    
    print("\n4. GEOGRAPHIC TARGETING INSIGHTS...")
    
    # District-level analysis
    district_analysis = df.groupby('District').agg({
        'predicted_vulnerable': ['count', 'sum'],
        'vulnerability_probability': 'mean',
        'HHIncome+Consumption+Residues/Day': 'mean',
        'HouseholdSize': 'mean'
    }).round(2)
    
    district_analysis.columns = ['Total_Households', 'Predicted_Vulnerable', 'Avg_Risk_Score', 'Avg_Income', 'Avg_HH_Size']
    district_analysis['Vulnerability_Rate'] = (district_analysis['Predicted_Vulnerable'] / district_analysis['Total_Households'] * 100).round(1)
    
    print("\nDistrict-Level Program Targeting:")
    print("District     | HH Count | Vulnerable | Rate  | Avg Risk | Avg Income")
    print("-" * 75)
    
    for district, row in district_analysis.sort_values('Vulnerability_Rate', ascending=False).iterrows():
        total = f"{int(row['Total_Households'])}"
        vulnerable = f"{int(row['Predicted_Vulnerable'])}"
        rate = f"{row['Vulnerability_Rate']:.1f}%"
        risk = f"{row['Avg_Risk_Score']:.2f}"
        income = f"${row['Avg_Income']:.2f}"
        print(f"{district:12} | {total:8} | {vulnerable:10} | {rate:5} | {risk:8} | {income:10}")
    
    # Risk category distribution by district
    print("\nRisk Category Distribution by District:")
    risk_by_district = pd.crosstab(df['District'], df['risk_category'], normalize='index') * 100
    print(risk_by_district.round(1))
    
    return district_analysis

def intervention_recommendations(df, feature_importance):
    """Generate intervention recommendations based on feature importance"""
    
    print("\n5. INTERVENTION RECOMMENDATIONS...")
    
    # Analyze top predictive features
    top_features = feature_importance.head(10)
    
    print("\nTop 10 Predictive Factors (Intervention Opportunities):")
    print("Rank | Feature                           | Importance | Intervention Type")
    print("-" * 80)
    
    # Map features to intervention types
    intervention_mapping = {
        'AgricValue': 'Agricultural productivity programs',
        'HouseholdIcome': 'Income generation initiatives',
        'income_per_capita': 'Household size planning/income support',
        'PerenialCropIncome': 'Long-term crop development',
        'agric_productivity': 'Agricultural efficiency training',
        'PersonalBusinessAndSelfEmployment': 'Business development support',
        'SeasonalCropIncome': 'Seasonal farming optimization',
        'CasualLabour': 'Skills training for stable employment',
        'AgricultureLand': 'Land access and management programs',
        'FormalEmployment': 'Job placement and skills development'
    }
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        feature = row['Feature']
        importance = row['Importance']
        intervention = intervention_mapping.get(feature, 'General household support')
        print(f"{i:4} | {feature:33} | {importance:10.3f} | {intervention}")
    
    # Generate specific recommendations by vulnerability level
    print(f"\nRecommendations by Risk Category:")
    
    for risk_cat in ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
        households = df[df['risk_category'] == risk_cat]
        if len(households) > 0:
            count = len(households)
            avg_income = households['HHIncome+Consumption+Residues/Day'].mean()
            avg_prob = households['vulnerability_probability'].mean()
            
            print(f"\n{risk_cat} ({count:,} households, avg risk: {avg_prob:.1%}):")
            
            if risk_cat == 'Critical Risk':
                print("  • Immediate cash transfers and emergency food assistance")
                print("  • Priority enrollment in all support programs")
                print("  • Monthly monitoring and support check-ins")
                
            elif risk_cat == 'High Risk':
                print("  • Targeted agricultural productivity programs")
                print("  • Business development loans and training")
                print("  • Quarterly monitoring")
                
            elif risk_cat == 'Medium Risk':
                print("  • Preventive agricultural extension services")
                print("  • Skills development programs")
                print("  • Bi-annual check-ins")
                
            else:  # Low Risk
                print("  • General community programs")
                print("  • Annual household surveys")
                print("  • Community-level infrastructure development")

def create_final_visualizations(df, district_analysis):
    """Create final summary visualizations"""
    
    print("\n6. CREATING FINAL VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RTV Household Vulnerability Assessment - Final Results', fontsize=16, fontweight='bold')
    
    # 1. Risk category distribution
    risk_counts = df['risk_category'].value_counts()
    colors = ['green', 'orange', 'red', 'darkred']
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Household Risk Distribution')
    
    # 2. Vulnerability by district
    district_vuln = district_analysis['Vulnerability_Rate'].sort_values(ascending=True)
    axes[0, 1].barh(range(len(district_vuln)), district_vuln.values, color='coral')
    axes[0, 1].set_yticks(range(len(district_vuln)))
    axes[0, 1].set_yticklabels(district_vuln.index)
    axes[0, 1].set_xlabel('Vulnerability Rate (%)')
    axes[0, 1].set_title('Vulnerability Rate by District')
    
    # 3. Prediction vs actual
    axes[0, 2].scatter(df['vulnerability_probability'], df['HHIncome+Consumption+Residues/Day'], 
                      c=df['actual_vulnerable'], cmap='RdYlGn_r', alpha=0.6)
    axes[0, 2].set_xlabel('Predicted Vulnerability Probability')
    axes[0, 2].set_ylabel('Daily Income (USD)')
    axes[0, 2].set_title('Model Predictions vs Actual Income')
    
    # 4. Income distribution by risk category
    for risk_cat in df['risk_category'].unique():
        if pd.notna(risk_cat):
            subset = df[df['risk_category'] == risk_cat]['HHIncome+Consumption+Residues/Day']
            axes[1, 0].hist(subset, alpha=0.6, label=risk_cat, bins=20)
    axes[1, 0].set_xlabel('Daily Income (USD)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Income Distribution by Risk Category')
    axes[1, 0].legend()
    
    # 5. Household size vs vulnerability
    hh_size_risk = df.groupby('HouseholdSize')['vulnerability_probability'].mean().reset_index()
    hh_size_risk = hh_size_risk[hh_size_risk['HouseholdSize'] <= 15]  # Filter outliers
    axes[1, 1].scatter(hh_size_risk['HouseholdSize'], hh_size_risk['vulnerability_probability'])
    axes[1, 1].set_xlabel('Household Size')
    axes[1, 1].set_ylabel('Average Vulnerability Probability')
    axes[1, 1].set_title('Household Size vs Vulnerability Risk')
    
    # 6. Intervention priority map
    intervention_data = df.groupby(['District', 'risk_category']).size().unstack(fill_value=0)
    intervention_data.plot(kind='bar', stacked=True, ax=axes[1, 2], color=colors)
    axes[1, 2].set_title('Intervention Priority by District')
    axes[1, 2].set_xlabel('District')
    axes[1, 2].set_ylabel('Number of Households')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].legend(title='Risk Category')
    
    plt.tight_layout()
    plt.savefig('final_vulnerability_assessment.png', dpi=300, bbox_inches='tight')
    print("✓ Saved final assessment visualization to 'final_vulnerability_assessment.png'")
    plt.close()

def save_final_results(df):
    """Save final results and predictions"""
    
    print("\n7. SAVING FINAL RESULTS...")
    
    # Save household predictions
    prediction_columns = [
        'HouseHoldID', 'District', 'Village', 'HouseholdSize', 
        'HHIncome+Consumption+Residues/Day', 'ProgressStatus',
        'predicted_vulnerable', 'vulnerability_probability', 'risk_category'
    ]
    
    predictions_df = df[prediction_columns].copy()
    predictions_df['recommendation'] = predictions_df['risk_category'].map({
        'Critical Risk': 'Immediate intervention required',
        'High Risk': 'Priority for targeted programs',
        'Medium Risk': 'Preventive support recommended',
        'Low Risk': 'General community programs'
    })
    
    predictions_df.to_csv('final_household_predictions.csv', index=False)
    print("✓ Saved household predictions to 'final_household_predictions.csv'")
    
    # Create summary report
    summary_stats = {
        'total_households': len(df),
        'predicted_vulnerable': df['predicted_vulnerable'].sum(),
        'vulnerability_rate': df['predicted_vulnerable'].mean(),
        'average_risk_score': df['vulnerability_probability'].mean(),
        'critical_risk_households': len(df[df['risk_category'] == 'Critical Risk']),
        'high_risk_households': len(df[df['risk_category'] == 'High Risk']),
        'medium_risk_households': len(df[df['risk_category'] == 'Medium Risk']),
        'low_risk_households': len(df[df['risk_category'] == 'Low Risk'])
    }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("HOUSEHOLD VULNERABILITY ASSESSMENT COMPLETED!")
    print("=" * 80)
    
    print(f"\nFinal Assessment Results:")
    print(f"• Total households assessed: {summary_stats['total_households']:,}")
    print(f"• Households requiring intervention: {summary_stats['predicted_vulnerable']:,} ({summary_stats['vulnerability_rate']:.1%})")
    print(f"• Critical risk households: {summary_stats['critical_risk_households']:,}")
    print(f"• High risk households: {summary_stats['high_risk_households']:,}")
    print(f"• Medium risk households: {summary_stats['medium_risk_households']:,}")
    print(f"• Low risk households: {summary_stats['low_risk_households']:,}")
    
    print(f"\nProgram Impact Potential:")
    total_high_critical = summary_stats['critical_risk_households'] + summary_stats['high_risk_households']
    print(f"• Immediate intervention target: {total_high_critical:,} households")
    print(f"• Program coverage potential: {(total_high_critical / summary_stats['total_households']) * 100:.1f}% of population")
    
    avg_daily_income = df['HHIncome+Consumption+Residues/Day'].mean()
    vulnerable_avg = df[df['predicted_vulnerable'] == 1]['HHIncome+Consumption+Residues/Day'].mean()
    income_gap = avg_daily_income - vulnerable_avg
    print(f"• Average income gap: ${income_gap:.2f}/day for vulnerable households")
    
    print(f"\nDeliverables Generated:")
    print(f"• final_household_predictions.csv - Individual household risk assessments")
    print(f"• final_vulnerability_assessment.png - Program planning visualizations")
    print(f"• best_vulnerability_model.pkl - Deployable prediction model")
    print(f"• All supporting analysis files and documentation")
    
    print(f"\nRecommended Next Steps:")
    print(f"1. Deploy model for real-time field officer assessments")
    print(f"2. Begin targeted interventions for {total_high_critical:,} high-risk households")
    print(f"3. Establish monitoring system for quarterly re-assessments")
    print(f"4. Scale model to additional regions")

def main():
    """Main execution function"""
    print("Starting comprehensive model evaluation...")
    
    # Load model and data
    model, df, model_info, feature_importance = load_model_and_data()
    
    if model is not None and df is not None:
        # Generate predictions for all households
        df = generate_predictions_for_all_households(model, df)
        
        # Analyze prediction accuracy
        analyze_prediction_accuracy(df)
        
        # Geographic insights
        district_analysis = geographic_insights(df)
        
        # Intervention recommendations
        intervention_recommendations(df, feature_importance)
        
        # Create visualizations
        create_final_visualizations(df, district_analysis)
        
        # Save final results
        save_final_results(df)
        
    else:
        print("✗ Failed to load required files. Please run previous scripts first.")

if __name__ == "__main__":
    main()
