# House-Price-Predictor

## Overview

This project aims to predict house prices based on various features such as location, square footage, number of bedrooms, and other factors. The goal is to develop a regression model that can accurately estimate house prices using different machine learning techniques, including OLS Regression, Random Forest, and XGBoost.

## Problem Statement

The goal of this project is to predict the sale price of houses based on various features such as area, number of bedrooms, furnishing status, proximity to the main road, and other relevant factors. The dataset provided includes several attributes of houses, with the primary objective being the prediction of housing prices using machine learning techniques.

## Dataset
The dataset used for this project is sourced from Kaggle.

You can access the dataset from the following link:
[Housing Prices Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data)

## Setup

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

print("Path to dataset files:", path)

df = pd.read_csv("/root/.cache/kagglehub/datasets/yasserh/housing-prices-dataset/versions/1/Housing.csv")

```

## Steps Taken

### 1. Data Preprocessing:
- Loaded the housing dataset and performed initial cleaning.

### 2. Outlier Removal:
- Detected and removed outliers in features by using the **IQR method** (Interquartile Range).
  - Calculated the **IQR** for each feature.
  - Defined outliers as values below the lower bound or above the upper bound of the IQR.
  - Removed rows with outliers to ensure cleaner data for model training.

### 3. Feature Engineering:
- Converted categorical features into numeric values using one-hot encoding (pd.get_dummies).

### 4. Model Training and Evaluation:
- Split the data into training and testing sets using train_test_split.
- Evaluated three different classification models:
  - OLS Regression
  - Random Forest Regressor
  - XGBoost Regressor
 
### 5. Residual Analysis:

- Visualized the residuals for all models using histograms and Q-Q plots.


### Conclusion
This project explores various regression models to predict house prices. The OLS Regression model performed the best among the three, achieving an R² score of 0.6078. Feature engineering and proper handling of outliers played a crucial role in improving model performance. Future improvements could involve more advanced feature selection techniques and hyperparameter tuning for better accuracy.

This project provides a solid foundation for house price prediction and can be extended with more data sources and advanced machine learning techniques.
