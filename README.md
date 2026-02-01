# 🏠 House Price Prediction using Random Forest Regressor

## 📌 Project Overview
This project focuses on predicting house prices using machine learning techniques. The dataset undergoes thorough preprocessing, feature engineering, and model training using a **Random Forest Regressor**. The model achieved a **Mean Absolute Error Percentage (MAEP) of 13.8%**, demonstrating reliable predictive performance.

---

## 📊 Dataset
The dataset contains various housing features such as:

- Property characteristics
- Location-based attributes
- Structural details
- Sale price (Target Variable)

*(Dataset source: Kaggle House Prices Dataset or similar real estate dataset)*

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:

### ✔ Handling Missing Values
- Imputed numerical missing values using statistical methods
- Handled categorical missing values appropriately

### ✔ Feature Encoding
- Converted categorical variables into numerical representations using encoding techniques

### ✔ Feature Scaling
- Applied scaling where required for model efficiency

### ✔ Outlier Handling
- Identified and handled extreme values to improve model stability

---

## 🤖 Model Training
### Model Used:
**Random Forest Regressor**

### Training Process:
- Dataset split into training and testing sets
- Model trained using optimized hyperparameters
- Performance evaluated using error metrics

---

## 📈 Model Performance
- **Mean Absolute Error Percentage (MAEP): 13.8%**

This indicates that the model predictions deviate approximately **13.8%** from actual house prices on average.

---

## 🧪 Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Absolute Error Percentage (MAEP)

---

## 📂 Project Structure
```
├── data
│   └── housing_dataset.csv
├── notebooks
│   └── preprocessing_and_training.ipynb
├── src
│   └── model_training.py
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook or Script
Open the Jupyter Notebook or run the Python script to train and test the model.

---

## 🔮 Future Improvements
- Hyperparameter tuning using GridSearchCV or Bayesian Optimization
- Trying advanced ensemble models (XGBoost, LightGBM)
- Deploying model as a web application
- Adding cross-validation
- Expanding feature engineering

---

## 📜 License
This project is open-source and available under the MIT License.

---

## 👤 Author
**Zain Ul Abideen**  
MPhil Statistics | Data Science & Machine Learning Enthusiast
