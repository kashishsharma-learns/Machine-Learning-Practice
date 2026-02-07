# Random Forest for Breast Cancer Detection 🩺🌲

This project implements a **Random Forest classifier** using **scikit-learn** to detect breast cancer based on diagnostic features.  
It includes **hyperparameter tuning**, **model evaluation**, **feature importance analysis**, and **model persistence**.

---

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features computed from digitized images of breast mass
- **Target labels:**
  - `0` → Malignant
  - `1` → Benign

---

## Model Pipeline

- **Algorithm:** Random Forest Classifier
- **Class Weight:** Balanced (to handle slight class imbalance)
- **Hyperparameter Optimization:** GridSearchCV
- **Scoring Metric:** ROC AUC
- **Cross-Validation:** 5-fold
- **Parallel Processing:** Enabled (`n_jobs=-1`)

---

## Hyperparameters Tuned

- `n_estimators`: 200, 400  
- `max_depth`: None, 8, 16  
- `min_samples_split`: 2, 5, 10  
- `min_samples_leaf`: 1, 2, 4  
- `max_features`: sqrt, log2, 0.5  
- `bootstrap`: True  

---

## Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- joblib

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib
