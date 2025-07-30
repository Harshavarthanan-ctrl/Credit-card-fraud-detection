# 💳 Credit Card Fraud Detection Using Machine Learning

This repository contains a **Credit Card Fraud Detection** model developed using **Python** and the **scikit-learn** library. The project leverages a supervised learning approach to accurately identify fraudulent transactions from anonymized credit card data.

---

## 📊 Dataset

The model is trained and tested using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- **Total Records**: 284,315 transactions  
- **Fraudulent Transactions**: 492 (~0.17%)  
- **Data Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> **Note:** The dataset features (V1–V28) are the result of **PCA transformation** to protect sensitive information. The `Amount` feature is normalized before model input.

---

## 🧠 Model Overview

- **Algorithm**: Random Forest Classifier  
- **Imbalanced Data Handling**: SMOTE (Synthetic Minority Oversampling Technique)  
- **Performance on Test Set**:
  - ✅ Accuracy: **99.98%**
  - 🎯 Precision: **99.97%**
  - 🔁 Recall: **1.0**

---

## 🚀 Deployment

The trained model is deployed using **Streamlit** to create an interactive web app.

### How it Works:
- Users input **29 features** (`V1–V28`, `Amount`)
- The model predicts whether the transaction is **fraudulent** or **legitimate**
- Friendly UI with a visualization image and fast response

---

---

## 📦 Requirements

- Python ≥ 3.7  
- scikit-learn  
- imbalanced-learn  
- pandas  
- numpy  
- Pillow  
- streamlit  

Install all dependencies with:

```bash
pip install -r requirements.txt
streamlit run App.py


