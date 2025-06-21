# Household Vulnerability Prediction Model

**RTV Senior Data Scientist Technical Assessment - Part A**

A comprehensive machine learning solution for predicting household vulnerability status based on socioeconomic indicators, enabling targeted interventions for rural development programs.

## 🎯 Project Overview

This project analyzes survey data from **3,897 households** across 4 districts in Uganda to predict vulnerability status and provide actionable insights for development programs. The solution achieves **99.0% accuracy** in identifying vulnerable households requiring immediate intervention.

### Key Results

- **42.5%** of households identified as vulnerable (struggling or severely struggling)
- **1,668 households** require immediate intervention
- **99.1% coverage rate** for vulnerable household identification
- **Logistic Regression** emerged as the best-performing model

## 📁 Project Structure

### Core Analysis Files

| File                                  | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| `01_data_exploration.py`              | Initial data quality assessment and basic statistics  |
| `02_target_variable_creation.py`      | Progress status engineering from income thresholds    |
| `03_comprehensive_eda.py`             | Detailed exploratory data analysis and visualizations |
| `04_ml_modeling.py`                   | Model training, comparison, and evaluation            |
| `05_model_evaluation_and_insights.py` | Advanced evaluation and business insights             |

### Main Notebook

- `household_vulnerability_analysis_complete.ipynb` - **Complete end-to-end analysis** combining all scripts with results and visualizations

### Model Assets

| File                                 | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `best_vulnerability_model_final.pkl` | Trained Logistic Regression model (primary) |
| `best_vulnerability_model.pkl`       | Alternative trained model                   |
| `model_features_info.csv`            | Feature information and specifications      |

### Data Files

| File                                            | Description                                       |
| ----------------------------------------------- | ------------------------------------------------- |
| `household_vulnerability_complete_analysis.csv` | **Main dataset** with predictions and risk scores |
| `data_with_proper_progress_status.csv`          | Dataset with engineered target variable           |
| `district_targeting_recommendations.csv`        | Geographic targeting strategy                     |
| `district_vulnerability_analysis.csv`           | District-level vulnerability metrics              |

### Analysis Results

| File                               | Description                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| `ANALYSIS_SUMMARY.md`              | **Executive summary** with key findings and recommendations |
| `analysis_summary_final.csv`       | Key metrics and model performance summary                   |
| `feature_importance_analysis.csv`  | Ranked feature importance for interventions                 |
| `feature_correlation_analysis.csv` | Feature correlation matrix                                  |
| `progress_status_summary.csv`      | Target variable distribution analysis                       |

### Visualizations

- `model_evaluation_summary.png` - Model performance dashboard
- `comprehensive_eda_analysis.png` - EDA visualization summary
- `progress_status_analysis.png` - Target variable analysis charts

### Configuration

- `requirements.txt` - Python dependencies
- `column_information.csv` - Data dictionary and column descriptions
- `comprehensive_eda_summary.txt` - EDA insights summary

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Complete Analysis

**Option 1: Jupyter Notebook (Recommended)**

```bash
jupyter notebook household_vulnerability_analysis_complete.ipynb
```

**Option 2: Individual Scripts**

```bash
# Run analysis pipeline step by step
python 01_data_exploration.py
python 02_target_variable_creation.py
python 03_comprehensive_eda.py
python 04_ml_modeling.py
python 05_model_evaluation_and_insights.py
```

### Load Pre-trained Model

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_vulnerability_model_final.pkl')

# Load processed data with predictions
df = pd.read_csv('household_vulnerability_complete_analysis.csv')

# Get predictions for new data
# predictions = model.predict(new_data)
```

## 📊 Key Findings

### Geographic Distribution

| District      | Households | Vulnerable | Vulnerability Rate |
| ------------- | ---------- | ---------- | ------------------ |
| **Mitooma**   | 986        | 507        | **51.4%**          |
| **Rubanda**   | 767        | 371        | **48.4%**          |
| **Rukungiri** | 1,319      | 609        | **46.2%**          |
| **Kanungu**   | 825        | 181        | **21.9%**          |

### Risk Segmentation

- **High Risk**: 1,625 households (41.7%) - _Immediate intervention required_
- **Medium Risk**: 149 households (3.8%) - _Preventive measures needed_
- **Low Risk**: 2,123 households (54.5%) - _Monitoring and resilience building_

### Model Performance

- **Accuracy**: 99.0%
- **Coverage**: 99.1% of vulnerable households identified
- **Precision**: 98.5% of predicted vulnerable households are actually vulnerable
- **Best Algorithm**: Logistic Regression

## 🎯 Top Risk Factors

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

## 💡 Intervention Recommendations

### Immediate Actions (High Risk - 1,625 households)

- Income support programs
- Emergency food assistance
- Rapid agricultural productivity interventions
- Microfinance and business development access

### Medium-term Programs (Medium Risk - 149 households)

- Preventive agricultural training
- Infrastructure improvements
- Financial literacy programs
- Business development support

### Long-term Development (Low Risk - 2,123 households)

- Resilience building programs
- Advanced agricultural techniques
- Market linkage development
- Leadership development opportunities

## 📈 Business Impact

### Program Targeting Efficiency

- **1,668 households** identified for immediate intervention
- **99.1% coverage** of truly vulnerable households
- **98.5% precision** in targeting (minimal false positives)
- **Cost-effective** resource allocation through risk-based segmentation

### Expected Outcomes

- Reduced household vulnerability through targeted interventions
- Improved program efficiency via data-driven decision making
- Enhanced monitoring capabilities using predictive analytics
- Scalable methodology for other regions/populations

## 🔄 Model Deployment

### Real-time Prediction

```python
# Example prediction workflow
import joblib
import pandas as pd

# Load model
model = joblib.load('best_vulnerability_model_final.pkl')

# Prepare new household data
new_household = pd.DataFrame({
    'AgricValue': [1500],
    'HouseholdIncome': [50],
    'income_per_capita': [0.8],
    # ... other features
})

# Get vulnerability prediction
vulnerability_prob = model.predict_proba(new_household)[:, 1]
is_vulnerable = model.predict(new_household)

print(f"Vulnerability Probability: {vulnerability_prob[0]:.2%}")
print(f"Requires Intervention: {'Yes' if is_vulnerable[0] else 'No'}")
```

### Monitoring and Maintenance

- **Annual retraining** with new survey data
- **Quarterly assessments** using the model
- **Performance tracking** and model drift detection
- **Feedback loop** from intervention outcomes

## 📋 Technical Details

### Data Processing

- **Target Variable**: Created from `HHIncome+Consumption+Residues/Day` using poverty thresholds
- **Feature Engineering**: Income per capita, infrastructure scores, asset ownership indicators
- **Missing Data**: 2.63% overall missing data handled through imputation
- **Validation**: Stratified cross-validation for robust performance estimates

### Model Selection

- **Algorithms Compared**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Business Metrics**: Coverage rate, targeting precision, intervention efficiency
- **Final Model**: Logistic Regression selected for optimal business impact

## 📞 Support

For questions about this analysis or implementation:

1. **Review Documentation**: Start with `ANALYSIS_SUMMARY.md` for executive summary
2. **Explore Data**: Use `household_vulnerability_complete_analysis.csv` for detailed household data
3. **Check Notebook**: Run `household_vulnerability_analysis_complete.ipynb` for full reproducible analysis
4. **Model Details**: Examine `feature_importance_analysis.csv` for intervention priorities

## 📄 License

This project is part of the RTV Senior Data Scientist Technical Assessment and contains analysis of household survey data for development program optimization.

---

**Status**: ✅ Analysis Complete - Ready for Implementation  
**Last Updated**: 2025  
**Author**: Cosmas Wamozo
