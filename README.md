# EcoNET Quality Assurance: Predicting Sensor Errors (CSC 522 Final Project)

This repository contains the complete implementation pipeline to detect true hardware failures within the deeply imbalanced EcoNET climate sensor dataset (3.6% anomaly rate). 

## Methodology Overview (Rubric Requirements)
* **Domain-Specific Novelty (Feature Engineering)**: The system models the temporal `temperature rate of change` (`temp_roc`) and the spatial deviations of sensor networks (`buddy_dev`).
* **Cost-Sensitive Threshold Optimization**: We sweep classification thresholds below standard 0.50, explicitly optimizing for the $F_2$ score to heavily penalize missing actual errors (Type II errors).
* **Metric Alignment**: Relies exclusively on PR-AUC and $F_2$ instead of standard baseline accuracy.

## Environment Setup (Google Colab)

Since the dataset contains over 6.5 million observations, we execute these models using Google Colab.

### 1. Data Mounting
Upload your EcoNET dataset `train.csv` (or similar chunk) to a folder in your Google Drive (e.g. `/MyDrive/EcoNET/train.csv`).

### 2. Dependencies
Ensure these are installed in your Colab runtime (they are pre-installed in default runtimes except `shap`):
```bash
pip install pandas numpy scikit-learn xgboost shap
```

### 3. Execution Order

Run the notebooks strictly in this order to generate models and perform evaluation:

1. **`baseline.ipynb`**: Demonstrates that naive prediction methods (`DummyClassifier`) fail via metrics despite high literal accuracy.
2. **`logistic_regression.ipynb`**: Trains the baseline linear model, utilizing `class_weight='balanced'` and generating 5-fold CV scores. 
3. **`random_forest.ipynb`**: Trains a non-linear tree model, generating robust parameter grids for the appendix.
4. **`xgboost_model.ipynb`**: **Primary Experiment.** This notebook explicitly tests our hypothesis. It trains two versions of gradient boosting models (Raw vs. Enhanced features) and proves that extracting physical `temp_roc` and `buddy_dev` metrics statistically improves F2 scores.
5. **`evaluation.ipynb`**: Run this final notebook to aggregate predictions. It prints precision-recall curves, dynamically sweeps for the optimal $F_2$ threshold, and renders the SHAP Summary plot proving feature validity.

## Project Structure
* `preprocessing.py` Contains the sklearn `ColumnTransformer`, imputation logic, Colab data mounters, and our custom functions that engineer domain-specific spatial/temporal logic securely before scaling occurs.
