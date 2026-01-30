# Paddy Yield Classification Project - Complete Deliverables

## 📋 Project Overview
This is a complete end-to-end machine learning classification project that predicts rice (paddy) yield categories (Low, Medium, High) based on agricultural, environmental, and management factors.

**Final Model Performance:**
- Test Accuracy: 87.81%
- F1-Score (Weighted): 87.55%
- Model: Tuned Random Forest Classifier

---

## 📁 Deliverables

### 1. Main Report
**File:** `Paddy_Yield_Classification_Report.docx`

Comprehensive 20+ page report including:
- Executive Summary
- Dataset Information (2,789 samples, 44 features)
- Problem Justification
  - Why this problem is important (food security, economic impact)
  - Why ML is necessary (complexity, non-linearity, scalability)
  - Literature review (3 research papers synthesized)
- Complete methodology and data preprocessing
- Model development and hyperparameter tuning
- Results with detailed evaluation metrics
- Error analysis
- Conclusions and recommendations
- References

### 2. Analysis Scripts

#### `paddy_classification.py`
The main classification pipeline that performs:
- Data loading and exploration
- Target variable creation (yield categorization)
- Data preprocessing (encoding, scaling)
- Train-validation-test split (64%-16%-20%)
- Baseline model (DummyClassifier)
- Training 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Hyperparameter tuning (GridSearchCV with 5-fold CV)
- Model evaluation and comparison
- Feature importance analysis
- Error analysis

**To run:** `python3 paddy_classification.py`

#### `generate_visualizations.py`
Creates all 7 required visualizations:
- Confusion matrix
- ROC curves (multiclass)
- Feature importance (top 20)
- Yield distribution by category
- Model performance comparison
- Feature analysis (top 6 features)
- Categorical features distribution

**To run:** `python3 generate_visualizations.py`

### 3. Visualizations (7 plots)

All plots are publication-quality (300 DPI) PNG files:

1. **plot1_confusion_matrix.png**
   - Shows classification performance across all three classes
   - Reveals that Medium yield is hardest to predict
   - High yield has 99% recall

2. **plot2_roc_curves.png**
   - ROC curves for all three classes (One-vs-Rest)
   - All AUC scores > 0.97 (excellent)
   - High yield: AUC = 0.995

3. **plot3_feature_importance.png**
   - Top 20 most important features from Random Forest
   - Top 5 features highlighted in coral
   - Nursery area phosphorus is most important (11.29%)

4. **plot4_yield_distribution.png**
   - Violin and box plots showing yield distribution by category
   - Clear separation between Low, Medium, and High
   - Sample sizes displayed

5. **plot5_model_comparison.png**
   - Compares all 5 models (Baseline + 4 ML models)
   - Shows both Accuracy and F1-Score
   - Demonstrates >2.5x improvement over baseline

6. **plot6_feature_analysis.png**
   - Box plots for top 6 most important features
   - Shows relationship between features and yield categories
   - Helps understand feature-target relationships

7. **plot7_categorical_features.png**
   - Distribution of categorical features by yield category
   - Shows variety, soil type, nursery method, and location effects
   - Stacked bar charts with percentages

---

## 🎯 Key Results Summary

### Model Performance
- **Baseline (Most Frequent):** 35.79% accuracy
- **Logistic Regression:** 87.92% accuracy
- **Random Forest:** 87.47% accuracy
- **Gradient Boosting:** 87.25% accuracy
- **Tuned Random Forest (Final):** 87.81% test accuracy

### Per-Class Performance
- **High Yield:** Precision 88%, Recall 99%, F1 93%
- **Low Yield:** Precision 87%, Recall 88%, F1 88%
- **Medium Yield:** Precision 88%, Recall 77%, F1 82%

### Top 5 Most Important Features
1. LP_nurseryarea (Lime Phosphorus in nursery) - 11.29%
2. DAP_20days (Diammonium Phosphate) - 9.28%
3. Seedrate (in Kg) - 9.17%
4. Urea_40Days - 9.00%
5. Hectares - 8.74%

---

## 📊 Dataset Details

**Source:** Paddy cultivation dataset from Tamil Nadu, India

**Constraints Met:**
✅ Instances: 2,789 (>1,000 required)
✅ Features: 44 (>20 required)
✅ Format: Tabular CSV
✅ Classification: 3 balanced classes
✅ Data quality: No missing values, no significant outliers

**Target Variable:**
- Low Yield: <17,508 Kg (31.0%)
- Medium Yield: 17,508-30,355 Kg (35.9%)
- High Yield: >30,355 Kg (33.1%)

**Feature Categories:**
- Agricultural management (9 features)
- Location & soil (3 features)
- Weather - rainfall (8 features)
- Weather - temperature (8 features)
- Weather - wind (8 features)
- Weather - humidity (4 features)
- Nursery management (4 features)

---

## 🔬 Methodology Highlights

### Data Preprocessing
- **Encoding:** Label Encoding for 8 categorical variables
- **Scaling:** StandardScaler for all features
- **Split:** 64% train, 16% validation, 20% test (stratified)
- **No missing values or outliers**
- **Balanced classes** (imbalance ratio: 1.16)

### Model Selection
- Tested multiple algorithms (linear, ensemble)
- Used validation set for model comparison
- Selected Random Forest for hyperparameter tuning
- GridSearchCV with 5-fold cross-validation

### Hyperparameter Tuning
- Optimized: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Best params: n_estimators=50, max_depth=10, min_samples_split=5
- Cross-validation F1-Score: 85.52%

### Evaluation Metrics
- Primary: Accuracy, F1-Score (weighted and macro)
- Per-class: Precision, Recall, F1-Score
- Visualization: Confusion matrix, ROC curves
- Feature analysis: Feature importance rankings

---

## 🚀 Practical Applications

1. **Early Warning System:** Identify at-risk fields during growing season
2. **Resource Allocation:** Prioritize extension services to low-yield predictions
3. **Policy Planning:** Inform food import/storage decisions
4. **Farm Management:** Guide late-season interventions

---

## 📚 Literature Context

The project builds on recent research:
- **Machine learning for rice yield:** Random Forest and Neural Networks most common
- **Data sources:** Satellite imagery, weather data, ground observations
- **Performance:** Recent studies achieve R² of 0.44-0.68 for regression tasks
- **This project:** Achieves 87.81% accuracy for classification (3 classes)

Key references synthesized:
1. Scientific Reports (2024) - SAR and optical remote sensing with ML
2. ScienceDirect (2025) - Deep learning with crop growth models
3. Computers & Electronics in Agriculture (2025) - Systematic review of 156 studies

---

## 💡 Key Insights

### Agricultural Insights
- **Nutrient management is critical:** Top features dominated by fertilizers and phosphorus
- **Early interventions matter:** Nursery management and 20-day fertilization highly predictive
- **Controllable factors dominate:** Management decisions more important than weather
- **Variety effects exist:** Different rice varieties show distinct yield patterns

### Machine Learning Insights
- **High performance achievable:** 87.81% accuracy with standard algorithms
- **Random Forest preferred:** Good balance of performance, interpretability, robustness
- **Medium yields are tricky:** Transitional zone harder to classify
- **Simple features sufficient:** No need for satellite imagery or complex sensors

---

## 🔄 How to Reproduce

1. **Run the main analysis:**
   ```bash
   python3 paddy_classification.py
   ```
   This will output full analysis results to console.

2. **Generate visualizations:**
   ```bash
   python3 generate_visualizations.py
   ```
   This creates all 7 plots as PNG files.

3. **Review the report:**
   Open `Paddy_Yield_Classification_Report.docx` for the complete write-up.

---

## 📝 Project Requirements Compliance

✅ **Dataset Selection:**
- From tabular format (CSV)
- 2,789 instances (>1,000 required)
- 44 features (>20 required)
- Classification task (3 classes)

✅ **Problem Justification:**
- Importance explained (food security, economic, optimization)
- ML necessity justified (complexity, non-linearity, scale)
- 3 research papers synthesized

✅ **Technical Requirements:**
- Used only: pandas, numpy, scikit-learn, matplotlib ✓
- No deep learning libraries ✓
- No AutoML ✓

✅ **Workflow Components:**
- Data understanding and preprocessing ✓
- Missing values, outliers, class balance handled ✓
- Categorical encoding (LabelEncoder) ✓
- Scaling with justification (StandardScaler) ✓
- Baseline model (DummyClassifier) ✓
- 3+ real models tested ✓
- Hyperparameter tuning (GridSearchCV) ✓
- Stratified train/val/test split ✓
- Multiple metrics (Accuracy, F1, Precision, Recall) ✓
- Confusion matrix ✓
- ROC curves ✓
- Feature importance analysis ✓
- Error analysis ✓
- 7 meaningful plots ✓

---

## 🎓 Conclusions

This project successfully demonstrates:
1. Machine learning can effectively predict rice yield categories
2. 87.81% accuracy represents practical, deployable performance
3. Nutrient management is the key controllable factor
4. Random Forest provides excellent balance of performance and interpretability
5. The model can support farmers, extension officers, and policymakers

The work contributes to food security by enabling proactive agricultural management and policy planning based on data-driven yield predictions.

---

## 📧 Project Information

**Project Type:** Machine Learning Classification
**Domain:** Agricultural Science / Food Security
**Dataset:** Paddy (Rice) Cultivation Data
**Region:** Tamil Nadu, India
**Date:** January 2026
