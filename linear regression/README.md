# Linear Regression on Diabetes Dataset 📉

This project demonstrates **Simple Linear Regression** using **scikit-learn** on the **Diabetes dataset**.  
The model predicts disease progression based on a single feature and visualizes the regression line.

---

## Dataset

- **Source:** `sklearn.datasets.load_diabetes`
- **Samples:** 442
- **Features Used:**  
  - One feature (index `2`) extracted from the dataset
- **Target:**  
  - Quantitative measure of disease progression one year after baseline

---

## Model Details

- **Algorithm:** Linear Regression
- **Feature Selection:** Single feature (univariate regression)
- **Train/Test Split:**  
  - Training: all but last 30 samples  
  - Testing: last 30 samples

---

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install numpy scikit-learn matplotlib
