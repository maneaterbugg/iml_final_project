# Paddy Yield Classification – Machine Learning Project

## Project Description
This project implements an end-to-end machine learning classification workflow to predict paddy (rice) yield categories (**Low**, **Medium**, **High**) using a tabular dataset from the UCI Machine Learning Repository.

The project focuses on comparing ensemble methods and performing extensive hyperparameter optimization to achieve high predictive accuracy.

## Repository Contents
* `paddy_yield_analysis.ipynb`: Main notebook containing the complete ML workflow.
* `paddydataset.csv`: Dataset used in the notebook.
* `requirements.txt`: List of required Python libraries.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/maneaterbugg/iml_final_project.git](https://github.com/maneaterbugg/iml_final_project.git)
    cd iml_final_project
    ```

2.  **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the notebook:**
    ```bash
    jupyter notebook paddy_yield_analysis.ipynb
    ```

4.  **Execute:** Run all cells to see the data processing, model tuning, and final evaluation.

## Dataset
* **Source:** UCI Machine Learning Repository.
* **Instances:** 2,789.
* **Features:** 44.
* **Target:** Paddy yield category (discretized into Low, Medium, High based on quantiles).
  
### Download Dataset
The dataset file is not included in this repository. Please download `paddydataset.csv` from the UCI Machine Learning Repository and place it in the **root directory** of this project (same folder as `paddy_yield_analysis.ipynb`).

After downloading, the repository structure should look like this:
- `paddy_yield_analysis.ipynb`
- `paddydataset.csv`
- `requirements.txt`

## Methodology Overview
1.  **Preprocessing:** Target discretization using 33rd and 67th percentiles to ensure balanced classes.
2.  **Splitting:** Stratified split into Training (64%), Validation (16%), and Test (20%) sets.
3.  **Baseline:** Comparison against a `DummyClassifier` (most frequent strategy).
4.  **Model Training:** Initial comparison of Logistic Regression, Random Forest, and Gradient Boosting.
5.  **Optimization:** Extensive **GridSearchCV** on the Random Forest model to tune `n_estimators`, `max_depth`, and other parameters.
6.  **Evaluation:** Final assessment on the independent test set using weighted F1-score and confusion matrices.

## Final Model Performance
* **Selected Model:** Tuned Random Forest Classifier.
* **Optimization:** Fine-tuned via 5-fold cross-validation.
* **Performance:** * The model achieved high accuracy and F1-scores, significantly outperforming the baseline.
    * Validated through error analysis and ROC/AUC visualizations.

## Notes
* **Libraries:** Built with `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.
* **Constraints:** No deep learning or AutoML; strictly followed course-approved methodologies.

---
**Author:** maneaterbugg
**Date:** January 2026
