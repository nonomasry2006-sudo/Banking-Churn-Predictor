# 🏦 Banking Churn Predictor

An end-to-end machine learning project that predicts whether a bank customer is likely to churn, served through an interactive Streamlit web app.

---

## Overview

Customer churn is one of the most important problems for retail banks — retaining an existing customer is far cheaper than acquiring a new one. This project:

1. Cleans and explores a dataset of 10,000 bank customers
2. Trains and compares classification models to predict churn
3. Serves the best model through an interactive Streamlit app

---

## Dataset

- **Source:** [Kaggle — Banking Churn Prediction Dataset](https://www.kaggle.com/datasets/juniyad/banking-churn-prediction-dataset)
- **Rows:** 10,000
- **Columns:** 14 (11 after cleaning)
- **Target:** `Exited` (binary: 0 = stayed, 1 = churned)
- **Class balance:** 80% stayed / 20% churned (imbalanced)

---

## Project Structure

```
ml_project/
├── Banking_Churn_Prediction_Dataset.csv   # original dataset
├── churn_cleaned.csv                       # cleaned dataset
├── train_model.ipynb                       # cleaning, EDA, preprocessing, modeling
├── churn_model.pkl                         # trained model (scikit-learn Pipeline)
├── model_columns.pkl                       # expected input columns for the model
├── app.py                                  # Streamlit web app
├── requirements.txt                        # Python dependencies
├── .gitignore                              # files to exclude from git
└── README.md
```

---

## Approach

### 1. Data Cleaning
- Dropped non-predictive identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Verified no missing values and no duplicate rows
- Confirmed numeric ranges were sensible and categorical values were clean

### 2. Exploratory Data Analysis

Key findings:
- Overall churn rate: **20%**
- Germany churns more than France and Spain
- Inactive members churn about **2× more** than active ones
- Age, Balance, and Geography showed visible differences between churners and non-churners

### 3. Preprocessing
- One-hot encoded categorical features (`Geography`, `Gender`) using `drop_first=True`
- 80/20 train/test split with `stratify=y` to preserve the churn ratio
- Standard scaling applied inside a `Pipeline` for the deployed model

### 4. Modeling

Compared two models, both with `class_weight="balanced"` to handle class imbalance:

| Model                 | ROC-AUC | Churn Precision | Churn Recall | Churn F1 |
|-----------------------|---------|-----------------|--------------|----------|
| Logistic Regression   | 0.68    | 0.30            | 0.62         | 0.40     |
| Random Forest         | 0.64    | 0.30            | 0.22         | 0.25     |

**Winner:** Logistic Regression (scaled + balanced). It caught ~62% of churners with a reasonable precision/recall trade-off.

### 5. Deployment
The final model is a scikit-learn `Pipeline` (StandardScaler → LogisticRegression) saved to `churn_model.pkl` and loaded by the Streamlit app for real-time predictions.

---

## Results Summary

- **ROC-AUC:** 0.68
- **Churn recall:** 62% — catches most actual churners
- **Churn precision:** 30% — about 1 in 3 flagged customers actually churns
- **Accuracy:** 63%

**Note on metrics:** Accuracy is intentionally not the headline metric here. The dataset is imbalanced, so a model that predicts "no churn" for everyone would achieve 80% accuracy while being useless. **Recall on the churn class** is the more relevant metric for this business problem.

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/Banking-Churn-Predictor.git
cd Banking-Churn-Predictor
```

### 2. Create the conda environment
```bash
conda create -n churn python=3.11 -y
conda activate churn
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### 4. (Optional) Retrain the model
Open `train_model.ipynb` in Jupyter or VS Code and run all cells. This regenerates `churn_model.pkl` and `model_columns.pkl`.

---

## Tech Stack

- **Python 3.11**
- **pandas** — data manipulation
- **scikit-learn** — modeling, preprocessing, metrics
- **matplotlib / seaborn** — EDA visualizations
- **Streamlit** — web app
- **joblib** — model serialization

---

## Author

**Noor Mohamed Elmasry**

---

## Acknowledgements

- Dataset: [Kaggle — Banking Churn Prediction Dataset](https://www.kaggle.com/datasets/juniyad/banking-churn-prediction-dataset)
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)

---

## License

This project is for educational purposes. The dataset belongs to its original Kaggle author.