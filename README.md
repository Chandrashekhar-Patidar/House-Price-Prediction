# 🏠 House Price Prediction

## 📌 Project Overview

This project uses **Machine Learning and Linear Regression** to predict house prices based on different property-related features.

The project demonstrates a basic end-to-end supervised machine learning workflow, including:

- Data loading
- Data exploration
- Missing-value handling
- Categorical data encoding
- Feature and target separation
- Train-test splitting
- Linear Regression
- Model prediction
- Model evaluation
- Feature scaling using StandardScaler

---

## 🎯 Objective

The objective of this project is to build a regression model that predicts the **SalePrice** of a house using available property features.

---

## 📊 Dataset

The project uses the `HousePricePrediction.csv` dataset.

### Dataset Information

- **Rows:** 2,919
- **Columns:** 13
- **Target Variable:** `SalePrice`

### Features

The dataset contains the following features:

| Feature | Description |
|---|---|
| `MSSubClass` | Type/class of dwelling |
| `MSZoning` | General zoning classification |
| `LotArea` | Lot size in square feet |
| `LotConfig` | Lot configuration |
| `BldgType` | Type of dwelling |
| `OverallCond` | Overall condition of the property |
| `YearBuilt` | Original construction year |
| `YearRemodAdd` | Remodeling year |
| `Exterior1st` | Exterior covering of the house |
| `BsmtFinSF2` | Finished basement area |
| `TotalBsmtSF` | Total basement area |
| `SalePrice` | House sale price — target variable |

The `Id` column is removed during preprocessing because it is an identifier rather than a useful predictive feature.

---

## 🔍 Exploratory Data Analysis

The notebook performs basic dataset exploration using:

```python
df.shape
df.head()
df.info()
df.describe()
```

These operations are used to understand:

- Dataset dimensions
- Data types
- Missing values
- Statistical properties of numerical features

---

## 🧹 Data Preprocessing

### 1. Removing ID

The `Id` column is removed because it is an identifier:

```python
df.drop(['Id'], axis=1, inplace=True)
```

### 2. Handling Missing Values

Missing values in `SalePrice` are filled using the mean:

```python
df['SalePrice'] = df['SalePrice'].fillna(
    df['SalePrice'].mean()
)
```

Remaining rows containing missing values are removed:

```python
df = df.dropna()
```

### 3. Categorical Encoding

Categorical features are converted into numerical features using **One-Hot Encoding**:

```python
cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

df = pd.get_dummies(
    df,
    columns=cols,
    drop_first=True
)
```

---

## 🎯 Feature and Target Selection

The target variable is:

```python
SalePrice
```

Features are separated from the target:

```python
x = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80:20 split**:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

- **80%** → Training data
- **20%** → Testing data
- `random_state=42` → Reproducible split

---

## 🤖 Model Used

### Linear Regression

The project uses `LinearRegression` from Scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

The trained model is then used to predict house prices:

```python
y_pred = model.predict(x_test)
```

---

## 📏 Model Evaluation

The model is evaluated using four regression metrics:

### R² Score

Measures how much of the variation in the target variable is explained by the model.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted house prices.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

### MAPE — Mean Absolute Percentage Error

Measures prediction error as a percentage.

### Baseline Results

| Metric | Result |
|---|---:|
| R² Score | **0.3469** |
| MAE | **32,690.89** |
| RMSE | **48,797.19** |
| MAPE | **0.2002** |

---

## ⚖️ Feature Scaling

The project also experiments with **StandardScaler** to standardize the features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
```

The Linear Regression model is then trained using the scaled data.

### Scaled Model Results

| Metric | Result |
|---|---:|
| R² Score | **0.3469** |
| MAE | **32,690.89** |
| RMSE | **48,797.19** |
| MAPE | **0.2002** |

In this experiment, feature scaling produced essentially the **same performance** as the unscaled Linear Regression model.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Jupyter Notebook**
- **Seaborn**

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── House_Price_Prediction.ipynb
├── HousePricePrediction.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Chandrashekhar-Patidar/House-Price-Prediction.git
```

### 2. Open the project

Open the project folder in **Jupyter Notebook or JupyterLab**.

### 3. Install required libraries

```bash
pip install pandas numpy scikit-learn seaborn
```

### 4. Run the notebook

Open:

```text
House_Price_Prediction.ipynb
```

Run the cells sequentially to reproduce the preprocessing, model training, predictions, and evaluation results.

---

## 🔮 Future Improvements

The notebook itself suggests improving the model further by trying:

- Other regression algorithms
- Bagging techniques
- Boosting techniques
- Additional feature engineering
- Better handling of missing values
- Hyperparameter tuning

These approaches could potentially improve the current Linear Regression performance.

---

## 👨‍💻 Author

**Chandrashekhar Patidar**

B.Tech — Artificial Intelligence & Machine Learning