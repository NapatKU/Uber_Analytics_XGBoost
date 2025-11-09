# Uber Analytics - Ride Completion Prediction (XGBoost)

This project focuses on analyzing Uber ride booking data and building a **classification model** using **XGBoost** to predict ride outcomes (Completed, Cancelled, or Incomplete). The goal is to understand key patterns behind cancellations and trip failures and to demonstrate how data-driven modeling can help improve ride completion rates in real-world ride-hailing systems.

---

## 📌 Dataset
The dataset used in this project is sourced from Kaggle and contains **150,000 booking records** with **21 features** representing customer behavior, driver information, ride conditions, timestamps, vehicle types, payment methods, and booking statuses.  
The data reflects the full booking pipeline from request to completion or cancellation.  

**Source:** [Uber Ride Analytics Dashboard Dataset](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)

---

## 🔍 Data Exploration & Preparation
### Key Data Characteristics:
- Contains **Numerical**, **Categorical**, and **Binary** features.
- Includes **missing values** due to conditional logging (e.g., cancellation reasons exist only when rides are canceled).
- Data is **class-imbalanced**, with most rides marked as *Completed*.

### Data Preprocessing Steps:
- Handle missing values (0 or “Unknown” where appropriate).
- Time formatting and feature type correction.
- One-Hot Encoding for categorical variables.
- SMOTE oversampling to address class imbalance.
- Scaling numerical features for model stability.

---

## 🤖 Model: XGBoost Classifier
We selected **XGBoost** due to its:
- High performance with tabular data
- Ability to handle complex relationships and nonlinear features
- Efficiency in training and inference

### Training Parameters:
| Parameter | Value |
|---------|--------|
| n_estimators | 50 |
| learning_rate | 0.1 |
| max_depth | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| random_state | 42 |

---

## ✅ Model Performance
| Metric | Result |
|--------|--------|
| **Accuracy** | **~96%** |

The model performs well overall, though some minority classes (e.g., rare cancellation cases) still show lower recall due to class imbalance — a known limitation and potential target for improvement.

---
