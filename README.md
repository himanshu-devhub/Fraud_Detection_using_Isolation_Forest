# Credit Card Fraud Detection using Isolation Forest

This project applies the **Isolation Forest** algorithm to detect **fraudulent transactions** in a credit card dataset. It uses **unsupervised anomaly detection** to flag outliers that may represent fraud.

---

## 🔍 Project Overview

- **Dataset**: `creditcard_2023.csv` (tabular data with features and a `Class` label)
- **Goal**: Detect anomalies (fraud cases) using the Isolation Forest algorithm
- **Libraries Used**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📁 Dataset

- The dataset should include a column named `Class` where:
  - `1` = Fraudulent transaction
  - `0` = Legitimate transaction
- The script automatically drops an `id` column if present.

---

## ⚙️ How It Works

1. **Preprocessing**:
   - Drop ID column (if present)
   - Split into features (`X`) and labels (`y`)
   - Standardize the feature set using `StandardScaler`

2. **Modeling**:
   - Use `IsolationForest` from `scikit-learn`
   - Detect anomalies without using the actual class labels

3. **Evaluation**:
   - Map model output to fraud predictions:
     - `-1` → Fraud (`1`)
     - `1` → Normal (`0`)
   - Compare against true labels using:
     - Confusion Matrix
     - Classification Report

4. **Visualization**:
   - Plot count of detected anomalies with `seaborn`

---

## 🧪 Sample Output

