# Credit Risk Decision Engine

An interactive machine learning application for credit default risk assessment, customer-level risk scoring, portfolio analysis, and model explainability — built with **Python, XGBoost, Scikit-learn, and Streamlit**.

## Overview

The Credit Risk Decision Engine transforms customer financial and repayment-history data into estimated default probabilities. It combines an XGBoost classification pipeline with an interactive dashboard to help users explore individual customer risk, analyze customer portfolios, and understand the factors influencing model predictions.

This project demonstrates an end-to-end machine learning workflow, from model evaluation and threshold optimization to deployment-oriented application development.

## Key Features

- **Customer Risk Assessment:** Estimate an individual customer's probability of default using financial and repayment-related features.
- **Portfolio Risk Scoring:** Upload a customer CSV to score multiple records and summarize portfolio-level risk indicators.
- **Threshold Optimization:** Adjust the classification threshold to balance precision and recall according to the intended analytical use case.
- **Model Performance Evaluation:** Review accuracy, precision, recall, F1-score, and ROC-AUC.
- **Feature Importance:** Identify the variables that contribute most to model predictions.
- **SHAP Explainability:** Explore how individual features influence model output.
- **Interactive Dashboard:** Navigate between overview, customer assessment, portfolio scoring, and model insights.

## Model Performance

The current model evaluation reports the following results:

| Metric | Result |
|---|---:|
| Model | XGBoost Classifier |
| ROC-AUC | 0.7782 |
| Selected classification threshold | 0.27 |
| Tuned validation precision | 0.5132 |
| Tuned validation recall | 0.5876 |
| Tuned validation F1-score | 0.5478 |

The classification threshold was tuned to improve the balance between precision and recall for the positive class. At the default threshold of 0.50, the model showed higher precision but lower recall in the reported evaluation.

These metrics describe the current model evaluation and are not a guarantee of performance on future or external data.

## Technology Stack

- **Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Explainable AI:** SHAP
- **Model Persistence:** Joblib and native XGBoost model serialization
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook / Google Colab, Visual Studio Code

## Application Modules

### 1. Overview
Displays model information and provides an entry point for portfolio-level analysis.

### 2. Customer Assessment
Accepts customer financial and repayment information and returns an estimated default probability with a threshold-based classification.

### 3. Portfolio Scoring
Accepts a CSV containing the required model input features and calculates portfolio-level risk indicators, including the number of customers flagged by the selected threshold.

### 4. Model Insights
Presents model performance metrics, feature importance rankings, and SHAP-based explanations to support model interpretability.

## Model Explainability

Feature importance and SHAP visualizations help investigate the relationship between customer characteristics and model predictions.

The current analysis identifies repayment-status variables such as `PAY_0` and `DELAYED_MONTHS_6M` among the most influential features. This emphasizes the importance of historical repayment behavior in the model's risk estimates.

Feature influence should be interpreted as a model relationship, not proof of causation.

## Installation and Setup

### Prerequisites

- Python 3.11 or another Python version compatible with the project's pinned dependencies
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/vis358/credit-risk-decision-engine.git
cd credit-risk-decision-engine
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is unavailable, use an installed version compatible with the project's dependencies and saved model artifacts.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Model Artifacts

The application requires the trained model and associated configuration files to be available in the expected project directory.

Typical artifacts include:

- `credit_risk_xgb_native.json` — native XGBoost model
- `credit_risk_metadata.json` — model metadata, feature definitions, and classification threshold
- `credit_risk_defaults.json` — default input values for customer assessment

Use the artifact filenames and formats expected by `app.py`. Ensure the native model loads successfully in the target environment before deploying.

## Example Workflow

1. Open the dashboard.
2. Navigate to Customer Assessment to evaluate an individual customer.
3. Review the estimated default probability and selected classification threshold.
4. Navigate to Portfolio Scoring and upload a CSV containing the required model input features.
5. Review portfolio-level risk indicators and scored records.
6. Explore Model Insights to understand model performance and influential features.

## Limitations and Responsible Use

This project is an **educational machine learning prototype**, not a validated lending decision system.

- Predicted probabilities are model estimates, not certainties.
- The selected threshold is a configurable classification choice, not an industry-standard lending cutoff.
- Model performance may change across populations and over time.
- Predictions should not be used alone to approve or reject real credit applications.
- Real-world deployment would require appropriate validation, fairness analysis, data protection, monitoring, and regulatory review.

## Future Enhancements

- Deploy the dashboard to Streamlit Community Cloud.
- Add downloadable portfolio scoring reports.
- Introduce probability calibration and drift monitoring.
- Expand fairness and subgroup performance analysis.
- Add automated model evaluation and reproducible training workflows.

## Author

**Vishnu Vardhan Solasa**

GitHub: [@vis358](https://github.com/vis358)

---

*Built as an end-to-end machine learning and analytics project exploring credit default prediction, model interpretability, and interactive risk analysis.*