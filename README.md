# Paddy Yield Classification – Machine Learning Project

## Project Description
This project implements an end-to-end machine learning classification workflow to predict paddy (rice) yield categories (**Low**, **Medium**, **High**) using a tabular dataset from the UCI Machine Learning Repository.

The entire analysis, modeling, and evaluation are contained in a single Jupyter Notebook.

## Repository Contents
* `paddy_yield_analysis.ipynb`: Main notebook containing the complete ML workflow.
* `paddydataset.csv`: Dataset used in the notebook.
* `requirements.txt`: List of required Python libraries.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2.  **Install required libraries (Optional but recommended):**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Open the notebook:**
    ```bash
    jupyter notebook paddy_yield_analysis.ipynb
    ```

4.  **Run the analysis:**
    * Inside Jupyter, select **Cell** > **Run All** to execute the workflow from start to finish.

## Dataset
* **Source:** UCI Machine Learning Repository
* **Instances:** 2,789
* **Features:** 44
* **Target:** Paddy yield category (Low / Medium / High)
* **Format:** Tabular CSV (included in repository)

## Methodology Overview
1.  **Preprocessing:** Data understanding, cleaning, and target discretization using quantile-based thresholds.
2.  **Splitting:** Stratified train/validation/test split.
3.  **Baseline:** Implementation of a `DummyClassifier` for performance benchmarking.
4.  **Modeling:** Training multiple models (Logistic Regression, Random Forest, Gradient Boosting).
5.  **Evaluation:** Model comparison using weighted F1-score and final evaluation on an independent test set.

## Final Model Performance
* **Selected Model:** Logistic Regression
* **Selection Criterion:** Validation weighted F1-score
* **Test Set Performance:**
    * **Accuracy:** ~ 0.88
    * **Weighted F1-score:** ~ 0.88

## Notes
* The notebook adheres strictly to the course project constraints.
* **Libraries:** Only standard libraries are used (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
* **Exclusions:** No deep learning or AutoML methods are included.

---
**Author:** Machine Learning Course Project – January 2026
