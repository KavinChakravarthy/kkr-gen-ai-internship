# House Price Prediction

This project predicts house prices using machine learning models on the California Housing dataset. The workflow includes data cleaning, feature engineering, preprocessing, and model training/evaluation, all demonstrated in the `model.ipynb` notebook.

## Table of Contents
- [House Price Prediction](#house-price-prediction)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Data](#data)
  - [Setup](#setup)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Preprocessing](#preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Modeling](#modeling)
  - [Results](#results)
  - [How to Run](#how-to-run)
  - [Project Structure](#project-structure)
- [Cats and Dogs Classification](#cats-and-dogs-classification)

## Project Overview
This project aims to build a robust regression pipeline to predict median house values in California. It demonstrates:
- Data cleaning and exploration
- Feature engineering (log transforms, ratios)
- Preprocessing (encoding, scaling)
- Model training (Linear Regression, Random Forest)
- Hyperparameter tuning (GridSearchCV)
- Model evaluation (R² score)

## Data
- The dataset is `housing.csv` (California Housing data), located in the `data/` directory.
- The target variable is `median_house_value`.

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `data/housing.csv` is present.

## Exploratory Data Analysis
- Data is loaded and basic info/stats are displayed.
- Missing values are dropped.
- Distributions and correlations are visualized using histograms and heatmaps.

## Preprocessing
- Log-transform is applied to right-skewed features: `total_rooms`, `total_bedrooms`, `population`, `households`.
- Categorical feature `ocean_proximity` is one-hot encoded.
- Data is split into train and test sets.
- StandardScaler is used to scale features.

## Feature Engineering
- New features are created:
  - `bedroom_ratio` = total_bedrooms / total_rooms
  - `rooms_per_household` = total_rooms / households
- (Optional) Other ratios can be added for experimentation.

## Modeling
- **Linear Regression**: Baseline model for comparison.
- **Random Forest Regressor**: Main ensemble model.
  - Trained on both raw and scaled features.
  - Hyperparameter tuning with GridSearchCV for best performance.

## Results
- Model performance is evaluated using R² score on the test set.
- Feature importances are visualized for Random Forest.
- The best model and its score are reported in the notebook.

## How to Run
1. Open `notebooks/model.ipynb` in Jupyter or VS Code.
2. Run all cells sequentially to reproduce the workflow and results.

## Project Structure
```
├── data/
│   └── housing.csv
├── notebooks/
│   └── model.ipynb
├── src
│   └── best_forest.joblib
│   └── forest.joblib
│   └── stacking_model1.joblib
│   └── stacking_model2.joblib
├── requirements.txt
```

# Cats and Dogs Classification
