# RTV Senior Data Scientist Technical Assessment - Part A

## Household Vulnerability Prediction Model - Analysis Summary

**Date:** 2025  
**Author:** Cosmas Wamozo  
**Project:** Household Vulnerability Prediction for Rural Targeting and Vulnerability Assessment

---

## 🎯 Executive Summary

This analysis successfully developed a machine learning model to predict household vulnerability based on demographic, economic, agricultural, and infrastructure factors. The model achieves **98% accuracy** and **97.2% F1 score**, enabling precise identification of vulnerable households for targeted interventions.

### Key Findings:

- **35.6%** of households are classified as vulnerable (struggling or severely struggling)
- **$2.13/day** income gap between vulnerable and non-vulnerable households
- **Region 2** shows highest vulnerability rate at **60.7%**
- Model identifies critical risk factors for intervention prioritization

---

## 📊 Dataset Overview

| Metric               | Value                                                   |
| -------------------- | ------------------------------------------------------- |
| **Total Households** | 500                                                     |
| **Total Features**   | 6,905 (original) → 40 (selected)                        |
| **Target Variable**  | Household vulnerability status                          |
| **Data Quality**     | High sparsity handled through careful feature selection |

### Target Variable Distribution:

- **On Track**: 267 households (53.4%)
- **At Risk**: 55 households (11.0%)
- **Struggling**: 66 households (13.2%)
- **Severely Struggling**: 112 households (22.4%)

**Binary Classification:**

- **Non-vulnerable**: 322 households (64.4%)
- **Vulnerable**: 178 households (35.6%)

---

## 🔬 Methodology

### 1. Data Exploration and Cleaning

- **Feature Categorization**: Organized 6,905 features into 8 categories
- **Missing Value Analysis**: Handled high sparsity (>90% missing in many features)
- **Target Variable Creation**: Engineered progress status from income thresholds

### 2. Feature Engineering

- **Income per capita**: Household income divided by household size
- **Household size categories**: Small, Medium, Large, Very Large
- **Asset ownership score**: Aggregated asset ownership indicators
- **Geographic encoding**: Region and district variables

### 3. Model Selection and Training

- **Algorithms Tested**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Best Model**: Gradient Boosting Classifier
- **Cross-validation**: 5-fold stratified cross-validation
- **Hyperparameter Optimization**: Grid search for optimal parameters

---

## 📈 Model Performance

### Binary Classification Results:

| Model                 | Accuracy  | Precision | Recall    | F1 Score  | AUC       |
| --------------------- | --------- | --------- | --------- | --------- | --------- |
| **Gradient Boosting** | **98.0%** | **97.2%** | **97.2%** | **97.2%** | **98.8%** |
| Logistic Regression   | 90.0%     | 86.5%     | 86.5%     | 86.5%     | 97.0%     |
| Random Forest         | 85.0%     | 78.3%     | 78.3%     | 78.3%     | 89.3%     |
| SVM                   | 74.0%     | 59.4%     | 59.4%     | 59.4%     | 84.5%     |

### Classification Report (Test Set):

```
                precision    recall  f1-score   support
Not Vulnerable       0.98      0.98      0.98        64
    Vulnerable       0.97      0.97      0.97        36
      accuracy                           0.98       100
```

---

## 🎯 Key Risk Factors

### Top 15 Most Important Features:

1. **Business_type_3** (57.4% importance)
2. **Categorical features** related to economic activities
3. **Business_number** - Number of household businesses
4. **Material_floor** - Housing quality indicator
5. **Land_own_start** - Land ownership timing
6. **Business_income_start** - Business income initiation

### Risk Factor Categories:

- **Economic**: Business ownership, income sources, VSLA participation
- **Agricultural**: Land ownership, crop diversity, livestock
- **Infrastructure**: Housing materials, water access, sanitation
- **Demographic**: Household size, age structure, education

---

## 🗺️ Geographic Analysis

### Vulnerability by Region:

- **Region 2**: 60.7% vulnerability rate (37/61 households)
- **Region 4**: 32.1% vulnerability rate (141/439 households)

**Regional Insights:**

- Region 2 requires immediate, intensive intervention
- Region 4 has larger absolute numbers but lower rate
- Geographic targeting essential for resource allocation

---

## 💰 Economic Analysis

### Income Patterns:

| Household Type     | Average Income/Day | Income Range   |
| ------------------ | ------------------ | -------------- |
| **Vulnerable**     | $1.12              | $0.13 - $1.77  |
| **Non-vulnerable** | $3.25              | $1.77 - $11.84 |
| **Income Gap**     | **$2.13**          | -              |

### Key Economic Indicators:

- **64.4%** of households earn below $2.15/day
- **35.6%** are in vulnerable categories (struggling/severely struggling)
- Strong correlation between business ownership and non-vulnerability

---

## 📋 Risk Segmentation

### Household Risk Levels:

| Risk Level        | Count | Percentage | Intervention Priority |
| ----------------- | ----- | ---------- | --------------------- |
| **Critical Risk** | 178   | 35.6%      | **Immediate**         |
| **Low Risk**      | 322   | 64.4%      | Monitoring            |
| **High Risk**     | 0     | 0.0%       | -                     |
| **Medium Risk**   | 0     | 0.0%       | -                     |

_Note: Current model produces binary classification (Critical/Low risk). Future iterations could implement more granular risk scoring._

---

## 💡 Actionable Recommendations

### 1. Immediate Interventions (Critical Risk - 178 households)

- **Cash transfer programs** for immediate relief
- **Emergency food assistance** for severely struggling households
- **Healthcare subsidies** for vulnerable families

### 2. Medium-term Programs (1-2 years)

- **Business development training** focusing on viable business types
- **Agricultural extension services** for improved farming practices
- **Housing improvement programs** targeting infrastructure upgrades
- **Financial inclusion** through VSLA expansion and microfinance

### 3. Long-term Development (2-5 years)

- **Education and skills training** for household members
- **Infrastructure development** in high-vulnerability regions
- **Market linkage programs** for agricultural products
- **Land tenure security** initiatives

### 4. Region-Specific Strategies

- **Region 2**: Intensive intervention package due to high vulnerability rate
- **Region 4**: Scaled intervention program given large population size

---

## 📊 Model Deployment Recommendations

### 1. Implementation Strategy

- **Quarterly assessments** using the model for monitoring
- **Real-time prediction** capability for new household data
- **Dashboard development** for program managers
- **Mobile app integration** for field officers

### 2. Model Maintenance

- **Annual retraining** with new data
- **Feature importance monitoring** for changing risk factors
- **Performance tracking** and model drift detection
- **Feedback loop** from intervention outcomes

### 3. Data Collection Priorities

- **Business activity details** (highest importance feature)
- **Housing quality indicators** for infrastructure planning
- **Land ownership patterns** for agricultural programs
- **Income sources diversity** for economic planning

---

## 📁 Deliverables

### Files Created:

1. **`01_data_exploration.py`** - Initial data analysis
2. **`02_target_variable_creation.py`** - Progress status engineering
3. **`03_comprehensive_eda.py`** - Detailed exploratory analysis
4. **`04_ml_modeling.py`** - Model training and evaluation
5. **`05_model_evaluation_and_insights.py`** - Advanced evaluation
6. **`model_ready_dataset.csv`** - Processed dataset for modeling
7. **`final_household_predictions.csv`** - Individual household predictions
8. **`feature_importance_analysis.csv`** - Detailed feature importance
9. **`model_evaluation_summary.png`** - Visualization dashboard

### Key Outputs:

- **Trained Model**: Gradient Boosting Classifier (98% accuracy)
- **Risk Predictions**: Individual household vulnerability scores
- **Feature Importance**: Ranked list of risk factors
- **Regional Analysis**: Vulnerability mapping by geographic area
- **Intervention Targets**: 178 critical risk households identified

---

## 🎉 Project Impact

### Immediate Value:

- **Precise targeting** of 178 most vulnerable households
- **Evidence-based resource allocation** using feature importance
- **Regional prioritization** with Region 2 identified for immediate action
- **Cost-effective interventions** through risk-based segmentation

### Expected Outcomes:

- **Reduced vulnerability** through targeted interventions
- **Improved program efficiency** via data-driven decision making
- **Enhanced monitoring capabilities** using predictive analytics
- **Scalable methodology** for other regions/populations

---

## 🔄 Next Steps

1. **Model Deployment**: Implement prediction system in operational environment
2. **Pilot Program**: Launch targeted interventions for critical risk households
3. **Impact Evaluation**: Measure intervention effectiveness using model predictions
4. **Scale-up Planning**: Expand successful interventions to broader population
5. **Model Enhancement**: Incorporate additional data sources and feedback

---

**Status: ✅ Analysis Complete - Ready for Implementation**

_This analysis provides a robust foundation for evidence-based household vulnerability interventions, enabling targeted support for the most at-risk families while optimizing resource allocation across the program area._
