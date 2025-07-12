

## Table of Contents
- [House Price Prediction](#house-price-prediction)
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
  - [Project Overview](#project-overview-1)
  - [Data](#data-1)
  - [Setup](#setup-1)
  - [Exploratory Data Analysis](#exploratory-data-analysis-1)
  - [Preprocessing](#preprocessing-1)
  - [Modeling](#modeling-1)
  - [Results](#results-1)
  - [How to Run](#how-to-run-1)
  - [Project Structure](#project-structure-1)
  - [Versions Used](#versions-used)

---

# House Price Prediction

This project predicts house prices using machine learning models on the California Housing dataset. The workflow includes data cleaning, feature engineering, preprocessing, and model training/evaluation, all demonstrated in the `model.ipynb` notebook.

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
- `StandardScaler` is used to scale features.

## Feature Engineering

New features are created:

- `bedroom_ratio` = total_bedrooms / total_rooms  
- `rooms_per_household` = total_rooms / households
- (Optional) Other ratios can be added for experimentation.

## Modeling

- **Linear Regression**: Baseline model for comparison.
- **Random Forest Regressor**: Main ensemble model.
  - Trained on both raw and scaled features.
  - Hyperparameter tuning with `GridSearchCV`.

## Results

- Model evaluated using R² score on the test set.
- Feature importances visualized.
- Best model score reported in the notebook.

## How to Run

1. Open `notebooks/model.ipynb` in Jupyter or VS Code.
2. Run all cells sequentially.

## Project Structure

```
├── data/
│   └── housing.csv
├── notebooks/
│   └── model.ipynb
├── src/
│   └── best_forest.joblib
│   └── forest.joblib
│   └── stacking_model1.joblib
│   └── stacking_model2.joblib
├── requirements.txt
```

# Cats and Dogs Classification

## Project Overview

This project builds a binary image classifier to distinguish between cats and dogs using a Convolutional Neural Network (CNN). It demonstrates:

- Image preprocessing and augmentation using `ImageDataGenerator`
- CNN model design and training with TensorFlow/Keras
- Saving the trained model for reuse
- Visualization of accuracy and loss

## Data

- 24,998 labeled images (cats and dogs) in a CSV:
  - `data_path`: path to image  
  - `label`: 0 for cat, 1 for dog  
- Images located in: `cats-and-dogs-classification/data/`
- Sample data link provided in `data_set_link.txt`

## Setup

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Exploratory Data Analysis

- Data loaded with Pandas.
- Class distribution plotted using Seaborn countplot.
- Image path-label alignment verified.
- Total image count confirmed.

## Preprocessing

- All images resized to 128x128 pixels.
- Pixel values normalized to [0, 1] scale.
- Data augmented with:
  - Rotation, zoom, width/height shift, shear, horizontal flip
- Processed using `ImageDataGenerator.flow_from_dataframe()`.

## Modeling

- **CNN Architecture**:
  - Conv2D(16) → MaxPool  
  - Conv2D(32) → MaxPool  
  - Conv2D(64) → MaxPool  
  - Flatten → Dense(512, relu) → Dense(1, sigmoid)
- Compiled with:
  - Loss: `binary_crossentropy`
  - Optimizer: `adam`
  - Metrics: `accuracy`
- Trained for 50 epochs using augmented data.

## Results

- Final training accuracy ≈ 84%
- Accuracy/loss visualizations plotted
- Trained model saved as `model_50.keras` in `src/`

## How to Run

1. Open `classifier.ipynb` in Jupyter or VS Code.
2. Run all cells to:
   - Preprocess image data
   - Train the CNN model
   - Visualize training progress

## Project Structure

```
├── cats-and-dogs-classification/
│   └── data/
│       ├── Cat/
│       └── Dog/
│   └── classifier.ipynb
├── src/
│   └── model_50.keras
├── requirements.txt
├── data_set_link.txt
```

## Versions Used

| Tool         | Version      |
|--------------|--------------|
| Python       | 3.12.3       |
| TensorFlow   | 2.19.0       |
| Keras        | via tf.keras |
| NumPy        | latest       |
| Matplotlib   | latest       |
| Pandas       | latest       |
| Seaborn      | latest       |