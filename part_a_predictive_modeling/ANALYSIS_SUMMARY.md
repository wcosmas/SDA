# RTV Senior Data Scientist Technical Assessment - Part A

## Household Vulnerability Prediction Model - Analysis Summary

**Date:** 2025  
**Author:** Cosmas Wamozo  
**Project:** Household Vulnerability Prediction for Rural Targeting and Vulnerability Assessment

---

## 🎯 Executive Summary

This analysis successfully developed a machine learning model to predict household vulnerability based on demographic, economic, agricultural, and infrastructure factors. The model achieves **99.0% accuracy** and **99.1% coverage rate**, enabling precise identification of vulnerable households for targeted interventions.

### Key Findings:

- **42.5%** of households are classified as vulnerable (struggling or severely struggling)
- **$2.24/day** average total daily income across all households
- **Mitooma District** shows highest vulnerability rate at **51.4%**
- Model identifies critical risk factors for intervention prioritization

---

## 📊 Dataset Overview

| Metric               | Value                                                 |
| -------------------- | ----------------------------------------------------- |
| **Total Households** | 3,897                                                 |
| **Total Features**   | 75 (original) → optimized selection                   |
| **Target Variable**  | Household vulnerability status                        |
| **Data Quality**     | 100% completeness, 2.63% missing data across features |

### Target Variable Distribution:

- **On Track**: 1,704 households (43.7%)
- **At Risk**: 535 households (13.7%)
- **Struggling**: 901 households (23.1%)
- **Severely Struggling**: 757 households (19.4%)

**Binary Classification:**

- **Non-vulnerable**: 2,239 households (57.5%)
- **Vulnerable**: 1,658 households (42.5%)

---

## 🔬 Methodology

### 1. Data Exploration and Cleaning

- **Feature Categorization**: Organized 75 features into meaningful categories
- **Missing Value Analysis**: Handled 2.63% missing data across the dataset
- **Target Variable Creation**: Engineered progress status from income thresholds

### 2. Feature Engineering

- **Income per capita**: Total daily income divided by household size
- **Infrastructure score**: Aggregated infrastructure access indicators
- **Asset ownership score**: Composite asset ownership indicators
- **Agricultural productivity**: Combined agricultural income metrics

### 3. Model Selection and Training

- **Algorithms Tested**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Best Model**: Logistic Regression
- **Cross-validation**: Stratified cross-validation approach
- **Hyperparameter Optimization**: Grid search for optimal parameters

---

## 📈 Model Performance

### Binary Classification Results:

| Model                   | Accuracy  | Coverage  | Precision | Business Impact  |
| ----------------------- | --------- | --------- | --------- | ---------------- |
| **Logistic Regression** | **99.0%** | **99.1%** | **98.5%** | **Best Overall** |
| Gradient Boosting       | 95.2%     | 94.8%     | 93.1%     | Good             |
| Random Forest           | 88.7%     | 87.3%     | 85.9%     | Moderate         |
| SVM                     | 82.4%     | 81.2%     | 79.8%     | Baseline         |

### Business Impact Metrics:

- **Vulnerable household coverage**: 99.1% (1,643/1,658)
- **Program targeting precision**: 98.5% (1,643/1,668)
- **Households requiring intervention**: 1,668

---

## 🎯 Key Risk Factors

### Top 10 Most Important Features:

1. **AgricValue** - Agricultural productivity indicator
2. **HouseholdIncome** - Primary household income
3. **Income per capita** - Per person daily income
4. **PerenialCropIncome** - Long-term crop income
5. **PersonalBusinessAndSelfEmployment** - Business ownership
6. **Infrastructure score** - Access to services
7. **CasualLabour** - Casual employment income
8. **Asset ownership score** - Household assets
9. **AgricultureLand** - Land ownership size
10. **SeasonalCropIncome** - Seasonal farming income

### Risk Factor Categories:

- **Economic**: Business ownership, income sources, employment status
- **Agricultural**: Land ownership, crop diversity, agricultural productivity
- **Infrastructure**: Access to water, healthcare, basic services
- **Demographic**: Household size, education, age structure

---

## 🗺️ Geographic Analysis

### Vulnerability by District:

- **Mitooma**: 51.4% vulnerability rate (507/986 households)
- **Rubanda**: 48.4% vulnerability rate (371/767 households)
- **Rukungiri**: 46.2% vulnerability rate (609/1,319 households)
- **Kanungu**: 21.9% vulnerability rate (181/825 households)

**District Insights:**

- Mitooma, Rubanda, and Rukungiri require intensive intervention
- Kanungu shows lower vulnerability but needs preventive measures
- Geographic targeting essential for resource allocation efficiency

---

## 💰 Economic Analysis

### Income Patterns:

| Metric                   | Value     | Range             |
| ------------------------ | --------- | ----------------- |
| **Average Daily Income** | $1.33/day | $0.01 - $7.05/day |
| **Average Total Income** | $2.24/day | $0.12 - $8.41/day |
| **Median Total Income**  | $1.96/day | -                 |
| **Consumption Average**  | $0.91/day | $0.00 - $7.06/day |

### Key Economic Indicators:

- **42.5%** of households are in vulnerable categories
- Strong correlation between agricultural productivity and non-vulnerability
- Infrastructure access significantly impacts vulnerability status

---

## 📋 Risk Segmentation

### Household Risk Levels:

| Risk Level      | Count | Percentage | Intervention Priority |
| --------------- | ----- | ---------- | --------------------- |
| **High Risk**   | 1,625 | 41.7%      | **Immediate**         |
| **Medium Risk** | 149   | 3.8%       | **Preventive**        |
| **Low Risk**    | 2,123 | 54.5%      | **Monitoring**        |

---

## 💡 Actionable Recommendations

### 1. Immediate Interventions (High Risk - 1,625 households)

- **Income support programs** for immediate relief
- **Emergency food assistance** for severely struggling households
- **Rapid agricultural productivity interventions**
- **Access to microfinance and business development**

### 2. Medium-term Programs (Medium Risk - 149 households)

- **Preventive agricultural training** and skills development
- **Infrastructure improvements** for better service access
- **Savings and financial literacy programs**
- **Business development support** for emerging entrepreneurs

### 3. Long-term Development (Low Risk - 2,123 households)

- **Resilience building programs** to maintain stability
- **Advanced agricultural techniques** and market linkages
- **Leadership development opportunities**
- **Community-based development initiatives**

### 4. District-Specific Strategies

- **Mitooma**: Comprehensive intervention package (highest vulnerability)
- **Rubanda & Rukungiri**: Scaled intervention programs
- **Kanungu**: Preventive measures and monitoring

---

## 📊 Model Deployment Recommendations

### 1. Implementation Strategy

- **Real-time prediction** capability for new household data
- **Quarterly assessments** using the model for monitoring
- **Dashboard development** for program managers
- **Mobile integration** for field officers

### 2. Model Maintenance

- **Annual retraining** with new survey data
- **Feature importance monitoring** for changing risk patterns
- **Performance tracking** and model drift detection
- **Feedback loop** from intervention outcomes

### 3. Data Collection Priorities

- **Agricultural productivity metrics** (highest importance)
- **Household income diversification** data
- **Infrastructure access indicators** for development planning
- **Asset ownership patterns** for wealth tracking

---

## 📁 Deliverables

### Files Created:

1. **`household_vulnerability_analysis_complete.ipynb`** - Complete analysis notebook
2. **`household_vulnerability_complete_analysis.csv`** - Enhanced dataset with predictions
3. **`best_vulnerability_model_final.pkl`** - Trained model for deployment
4. **`analysis_summary_final.csv`** - Key metrics summary
5. **`district_targeting_recommendations.csv`** - Geographic targeting strategy

### Key Outputs:

- **Trained Model**: Logistic Regression (99.0% accuracy)
- **Risk Predictions**: Individual household vulnerability scores and categories
- **Geographic Targeting**: District-level intervention priorities
- **Intervention Strategy**: 1,668 households identified for immediate support

---

## 🎉 Project Impact

### Immediate Value:

- **Precise targeting** of 1,668 most vulnerable households
- **Evidence-based resource allocation** using feature importance
- **District prioritization** with Mitooma identified for immediate action
- **Cost-effective interventions** through risk-based segmentation

### Expected Outcomes:

- **Reduced vulnerability** through targeted interventions
- **Improved program efficiency** via data-driven decision making
- **Enhanced monitoring capabilities** using predictive analytics
- **Scalable methodology** for other regions/populations

---

**Status: ✅ Analysis Complete - Ready for Implementation**

_This analysis provides a robust foundation for evidence-based household vulnerability interventions, enabling targeted support for the most at-risk families while optimizing resource allocation across the program area._
