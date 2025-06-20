#!/usr/bin/env python3
"""
RTV Senior Data Scientist Technical Assessment - Part A
04. Machine Learning Modeling

This script develops and evaluates ML models for household vulnerability prediction
using the cleaned and analyzed dataset.

Author: Cosmas Wamozo
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset for modeling"""
    
    print("=" * 80)
    print("RTV HOUSEHOLD VULNERABILITY ASSESSMENT - ML MODELING")
    print("=" * 80)
    
    try:
        print("\n1. LOADING PROCESSED DATASET...")
        df = pd.read_csv('data_with_proper_progress_status.csv')
        print(f"✓ Successfully loaded dataset: {df.shape[0]} households, {df.shape[1]} variables")
        
        # Verify ProgressStatus exists
        if 'ProgressStatus' not in df.columns:
            print("✗ ProgressStatus variable not found. Please run previous scripts first.")
            return None, None
            
        # Create binary target variable for vulnerability
        df['is_vulnerable'] = df['ProgressStatus'].isin(['Struggling', 'Severely Struggling']).astype(int)
        
        print(f"✓ Target variable distribution:")
        target_dist = df['is_vulnerable'].value_counts()
        for target, count in target_dist.items():
            label = 'Vulnerable' if target == 1 else 'Non-vulnerable'
            percentage = count / len(df) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        return df, df.columns.tolist()
        
    except FileNotFoundError:
        print("✗ Processed dataset not found. Please run previous scripts first.")
        return None, None

def select_and_engineer_features(df):
    """Select and engineer features for modeling"""
    
    print("\n2. FEATURE SELECTION AND ENGINEERING...")
    
    # Define feature categories for modeling
    numeric_features = [
        'HouseholdSize', 'TimeToOPD', 'TimeToWater', 'AgricultureLand',
        'Season1CropsPlanted', 'Season2CropsPlanted', 'PerennialCropsGrown',
        'Season1VegetableIncome', 'Season2VegatableIncome', 'VegetableIncome',
        'FormalEmployment', 'PersonalBusinessAndSelfEmployment', 'CasualLabour',
        'RemittancesAndGifts', 'RentIncome', 'SeasonalCropIncome',
        'PerenialCropIncome', 'LivestockIncome', 'AgricValue',
        'HouseholdIcome', 'Assets.1'  # Using Assets.1 instead of Assets to avoid large numbers
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
    
    print(f"Selected features:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Binary features: {len(binary_features)}")
    
    # Feature engineering
    print(f"\n3. FEATURE ENGINEERING...")
    
    # Create derived features
    if 'HouseholdSize' in df.columns and 'HouseholdIcome' in df.columns:
        df['income_per_capita'] = df['HouseholdIcome'] / df['HouseholdSize'].replace(0, 1)
        numeric_features.append('income_per_capita')
        print("  ✓ Created income_per_capita")
    
    if 'AgricultureLand' in df.columns and 'AgricValue' in df.columns:
        df['agric_productivity'] = df['AgricValue'] / (df['AgricultureLand'].replace(0, 0.1))
        numeric_features.append('agric_productivity')
        print("  ✓ Created agric_productivity")
    
    # Create household size categories
    if 'HouseholdSize' in df.columns:
        df['household_size_category'] = pd.cut(df['HouseholdSize'], 
                                             bins=[0, 3, 5, 7, float('inf')], 
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])
        categorical_features.append('household_size_category')
        print("  ✓ Created household_size_category")
    
    # Create asset ownership score
    asset_cols = ['radios_owned', 'phones_owned']
    available_asset_cols = [col for col in asset_cols if col in df.columns]
    if available_asset_cols:
        df['asset_ownership_score'] = df[available_asset_cols].sum(axis=1)
        numeric_features.append('asset_ownership_score')
        print("  ✓ Created asset_ownership_score")
    
    # Create infrastructure access score
    infra_cols = ['latrine_constructed', 'tippy_tap_available', 'soap_ash_available',
                  'bathroom_constructed', 'kitchen_house']
    available_infra_cols = [col for col in infra_cols if col in df.columns]
    if available_infra_cols:
        df['infrastructure_score'] = df[available_infra_cols].sum(axis=1)
        numeric_features.append('infrastructure_score')
        print("  ✓ Created infrastructure_score")
    
    all_features = numeric_features + categorical_features + binary_features
    print(f"\nTotal features for modeling: {len(all_features)}")
    
    return df, numeric_features, categorical_features, binary_features

def create_preprocessing_pipeline(numeric_features, categorical_features, binary_features):
    """Create preprocessing pipeline for the features"""
    
    print(f"\n4. CREATING PREPROCESSING PIPELINE...")
    
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
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ])
    
    print(f"✓ Created preprocessing pipeline")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Binary features: {len(binary_features)}")
    
    return preprocessor

def train_and_evaluate_models(X, y, preprocessor):
    """Train and evaluate multiple ML models"""
    
    print(f"\n5. MODEL TRAINING AND EVALUATION...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    print(f"\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  Cross-validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  Test Precision: {precision:.3f}")
        print(f"  Test Recall: {recall:.3f}")
        print(f"  Test F1-score: {f1:.3f}")
        print(f"  Test AUC: {auc:.3f}")
        
        results[name] = {
            'pipeline': pipeline,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = name
    
    print(f"\n✓ Best performing model: {best_model} (F1-score: {best_score:.3f})")
    
    return results, X_test, y_test, best_model

def analyze_feature_importance(results, numeric_features, categorical_features, binary_features):
    """Analyze feature importance for tree-based models"""
    
    print(f"\n6. FEATURE IMPORTANCE ANALYSIS...")
    
    # Get feature importance from Random Forest and Gradient Boosting
    tree_models = ['Random Forest', 'Gradient Boosting']
    
    for model_name in tree_models:
        if model_name in results:
            print(f"\n{model_name} Feature Importance:")
            
            pipeline = results[model_name]['pipeline']
            model = pipeline.named_steps['classifier']
            
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps['preprocessor']
            
            # Get feature names
            feature_names = []
            
            # Numeric features
            feature_names.extend(numeric_features)
            
            # Categorical features (after one-hot encoding)
            if categorical_features:
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                feature_names.extend(cat_features)
            
            # Binary features
            feature_names.extend(binary_features)
            
            # Get importance scores
            importance_scores = model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            # Display top 15 features
            print("Top 15 most important features:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"  {i:2d}. {row['Feature'][:40]:40} {row['Importance']:.4f}")
            
            # Save feature importance
            if model_name == 'Random Forest':  # Save for the Random Forest model
                importance_df.to_csv('feature_importance_analysis.csv', index=False)
                print(f"\n✓ Saved feature importance to 'feature_importance_analysis.csv'")

def create_model_visualizations(results, X_test, y_test, best_model):
    """Create visualizations for model performance"""
    
    print(f"\n7. CREATING MODEL VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model comparison (F1 scores)
    model_names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in model_names]
    auc_scores = [results[name]['auc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, f1_scores, width, label='F1-score', alpha=0.8)
    axes[0, 0].bar(x + width/2, auc_scores, width, label='AUC', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Confusion matrix for best model
    best_results = results[best_model]
    cm = confusion_matrix(y_test, best_results['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    axes[0, 1].set_xticklabels(['Non-vulnerable', 'Vulnerable'])
    axes[0, 1].set_yticklabels(['Non-vulnerable', 'Vulnerable'])
    
    # 3. ROC curves for all models
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Prediction probability distribution for best model
    y_pred_proba = best_results['y_pred_proba']
    
    vulnerable_proba = y_pred_proba[y_test == 1]
    non_vulnerable_proba = y_pred_proba[y_test == 0]
    
    axes[1, 1].hist(non_vulnerable_proba, bins=20, alpha=0.6, label='Non-vulnerable', density=True)
    axes[1, 1].hist(vulnerable_proba, bins=20, alpha=0.6, label='Vulnerable', density=True)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'Prediction Probability Distribution - {best_model}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved model evaluation visualization to 'model_evaluation_summary.png'")
    plt.close()

def save_best_model_and_results(results, best_model, numeric_features, categorical_features, binary_features):
    """Save the best model and create final results"""
    
    print(f"\n8. SAVING BEST MODEL AND RESULTS...")
    
    # Save the best model
    best_pipeline = results[best_model]['pipeline']
    joblib.dump(best_pipeline, 'best_vulnerability_model.pkl')
    print(f"✓ Saved best model ({best_model}) to 'best_vulnerability_model.pkl'")
    
    # Create model summary
    model_summary = {
        'best_model': best_model,
        'model_performance': {
            'accuracy': results[best_model]['accuracy'],
            'precision': results[best_model]['precision'],
            'recall': results[best_model]['recall'],
            'f1_score': results[best_model]['f1'],
            'auc': results[best_model]['auc']
        },
        'feature_counts': {
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'binary_features': len(binary_features),
            'total_features': len(numeric_features) + len(categorical_features) + len(binary_features)
        },
        'all_model_results': {
            name: {
                'accuracy': results[name]['accuracy'],
                'precision': results[name]['precision'],
                'recall': results[name]['recall'],
                'f1_score': results[name]['f1'],
                'auc': results[name]['auc']
            } for name in results.keys()
        }
    }
    
    # Save model information
    model_info_df = pd.DataFrame([
        ['Model Type', best_model],
        ['Accuracy', f"{results[best_model]['accuracy']:.3f}"],
        ['Precision', f"{results[best_model]['precision']:.3f}"],
        ['Recall', f"{results[best_model]['recall']:.3f}"],
        ['F1-Score', f"{results[best_model]['f1']:.3f}"],
        ['AUC', f"{results[best_model]['auc']:.3f}"],
        ['Total Features', len(numeric_features) + len(categorical_features) + len(binary_features)],
        ['Numeric Features', len(numeric_features)],
        ['Categorical Features', len(categorical_features)],
        ['Binary Features', len(binary_features)]
    ], columns=['Metric', 'Value'])
    
    model_info_df.to_csv('model_features_info.csv', index=False)
    print("✓ Saved model information to 'model_features_info.csv'")
    
    # Print final summary
    print(f"\n" + "=" * 80)
    print("MACHINE LEARNING MODELING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\nBest Model Performance ({best_model}):")
    print(f"• Accuracy: {results[best_model]['accuracy']:.1%}")
    print(f"• Precision: {results[best_model]['precision']:.1%}")
    print(f"• Recall: {results[best_model]['recall']:.1%}")
    print(f"• F1-Score: {results[best_model]['f1']:.1%}")
    print(f"• AUC: {results[best_model]['auc']:.1%}")
    
    print(f"\nModel Features:")
    print(f"• Total features used: {len(numeric_features) + len(categorical_features) + len(binary_features)}")
    print(f"• Numeric features: {len(numeric_features)}")
    print(f"• Categorical features: {len(categorical_features)}")
    print(f"• Binary features: {len(binary_features)}")
    
    print(f"\nAll Model Comparison:")
    for name, result in results.items():
        marker = "👑" if name == best_model else "  "
        print(f"{marker} {name}: F1={result['f1']:.3f}, AUC={result['auc']:.3f}")
    
    print(f"\nFiles Generated:")
    print(f"• best_vulnerability_model.pkl - Trained model ready for deployment")
    print(f"• model_features_info.csv - Model configuration and performance")
    print(f"• feature_importance_analysis.csv - Feature importance rankings")
    print(f"• model_evaluation_summary.png - Performance visualizations")
    
    print(f"\nNext Steps:")
    print(f"1. Run '05_model_evaluation_and_insights.py' for detailed evaluation")
    print(f"2. Use the model for predictions on new data")
    print(f"3. Deploy the model for field officer use")

def main():
    """Main execution function"""
    print("Starting ML modeling...")
    
    # Load and prepare data
    df, all_columns = load_and_prepare_data()
    
    if df is not None:
        # Feature selection and engineering
        df, numeric_features, categorical_features, binary_features = select_and_engineer_features(df)
        
        # Prepare features and target
        all_features = numeric_features + categorical_features + binary_features
        X = df[all_features]
        y = df['is_vulnerable']
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features, binary_features)
        
        # Train and evaluate models
        results, X_test, y_test, best_model = train_and_evaluate_models(X, y, preprocessor)
        
        # Analyze feature importance
        analyze_feature_importance(results, numeric_features, categorical_features, binary_features)
        
        # Create visualizations
        create_model_visualizations(results, X_test, y_test, best_model)
        
        # Save best model and results
        save_best_model_and_results(results, best_model, numeric_features, categorical_features, binary_features)
        
    else:
        print("✗ Failed to load data. Please run previous scripts first.")

if __name__ == "__main__":
    main() 