# Credit Risk Prediction Model

This project implements a machine learning pipeline to classify credit risk using the Statlog (German Credit Data). The model is designed to assist financial institutions in identifying high-risk loan applicants with high sensitivity.

## 🚀 Major Milestones & Key Visualizations

### 1. Data Insights & Distribution
Extensive Exploratory Data Analysis (EDA) was conducted to identify the most predictive features. We found that checking account status and duration are the strongest indicators of risk.

<img width="751" height="395" alt="image" src="https://github.com/user-attachments/assets/baadd153-07c6-4e39-8877-6885e23c4de2" />


### 2. Risk Feature Correlation
We analyzed how numerical variables like credit duration impact the likelihood of default. The density plots revealed a clear trend: as loan duration increases, the concentration of "Bad Risk" cases rises significantly.

<img width="473" height="359" alt="image" src="https://github.com/user-attachments/assets/99ce7fc0-b12c-4f8f-9a8f-63ddb4d36d12" />

### 3. High-Recall Model Performance
A Decision Tree Classifier was optimized to prioritize the detection of "Bad" risks. By adjusting class weights and thresholds, we achieved a significant recall milestone to minimize financial loss from false approvals.

* **Key Metric:** **0.83 Recall** for the "Bad Risk" class.
* **Optimization:** Utilized Cost-Complexity Pruning (`ccp_alpha`) to balance accuracy and generalization.

<img width="890" height="682" alt="image" src="https://github.com/user-attachments/assets/df4c475d-5d02-4778-abca-dbc527c802bd" />

### 4. Model Interpretability (SHAP)
To move beyond a "black box" approach, SHAP (SHapley Additive exPlanations) was integrated to explain why the model flags specific individuals as high risk.

<img width="1033" height="893" alt="image" src="https://github.com/user-attachments/assets/103d97a0-8c7b-4ed2-a69b-6e06fdf47f76" />



<img width="784" height="731" alt="image" src="https://github.com/user-attachments/assets/87f6dda8-67f8-40da-9d1e-1d3185a45652" />

## 🛠️ Tech Stack
* **Analysis:** Python, Pandas, NumPy
* **Modeling:** Scikit-learn (Decision Trees)
* **Visualization:** Matplotlib, Seaborn
* **Explainability:** SHAP

## 📂 Dataset Information
The dataset used is the **Statlog (German Credit Data)** from the UCI Machine Learning Repository, containing 1,000 instances with 20 categorical/symbolic attributes.
