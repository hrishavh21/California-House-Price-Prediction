# California House Price Predictor

A machine learning web application that estimates the median house value for a California district from socioeconomic, household, and geographical features. The application is built with **Streamlit** and uses a trained **HistGradientBoosting** model.

> **Disclaimer:** Predictions are model estimates based on the California Housing dataset. They are not official property valuations and should not be used as the sole basis for a real-estate decision.

## Features

- Interactive inputs for eight California district features.
- Estimated median house value displayed in US dollars.
- Interactive map showing the selected latitude and longitude.
- Model training and evaluation workflow.
- Exploratory data analysis (EDA) visualizations.

## Tech stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Project structure

```text
California-House-Price-Prediction/
├── california_app.py
├── california_housing_pipeline.py
├── california_housing_model.pkl
├── requirements.txt
├── README.md
├── outputs/
│   ├── 1_histograms_skewness.png
│   ├── 2_boxplots_distribution.png
│   ├── 3_correlation_heatmap.png
│   ├── 4_scatter_medinc_vs_target.png
│   └── actual_vs_predicted.png
└── screenshots/
    ├── home_page.png
    └── prediction_result.png
```

The `outputs/` and `screenshots/` folders are suggested locations for generated charts and screenshots. Add them after creating the files. The trained model file is required to run the app.

## Dataset

The training script loads the **California Housing dataset** through `sklearn.datasets.fetch_california_housing`.

The model uses these eight input features:

| Feature | Meaning |
|---|---|
| `MedInc` | Median income in the district |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | District block population |
| `AveOccup` | Average household occupancy |
| `Latitude` | District latitude |
| `Longitude` | District longitude |

The target is `MedHouseVal`, the median house value represented by the dataset in units of $100,000.

## Machine learning workflow

The training script performs the following steps:

1. Loads the California Housing dataset into a Pandas DataFrame.
2. Checks the data structure, missing values, duplicate rows, and feature correlations.
3. Creates histograms, box plots, a correlation heatmap, and a median-income scatter plot.
4. Filters target values at or above `5.0` and applies IQR-based filtering to `AveRooms`, `AveBedrms`, `AveOccup`, and `Population`.
5. Splits the data into training and test sets (80% / 20%, with `random_state=42`).
6. Trains and compares Linear Regression, Random Forest, and HistGradientBoosting models.
7. Uses `RandomizedSearchCV` with 5-fold cross-validation to tune a HistGradientBoosting model.
8. Evaluates the tuned model using MAE, MSE, RMSE, and R².
9. Saves the fitted `RandomizedSearchCV` object as `california_housing_model.pkl` using Joblib.

The training script prints the evaluation metrics and best parameters. Add the actual values from your run here when you have them; they are not hard-coded in this documentation.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL and folder name with your GitHub repository details.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Make sure `california_housing_model.pkl` is in the project root, alongside `california_app.py`.

```bash
streamlit run california_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) that you can open in your browser.

## Train the model again

The uploaded training script was named `import matplotlib.py`. Rename it to `california_housing_pipeline.py` before using the command below, so that it matches the filename referenced by the app.

Create the output folder first because the training script saves plots inside it:

```bash
mkdir outputs
python california_housing_pipeline.py
```

The first dataset fetch may need an internet connection. The script saves the trained model as `california_housing_model.pkl` in the project root.

## Model evaluation

The training script reports:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination

Use the values printed by your own training run to complete this table:

| Metric | Result |
|---|---:|
| MAE | Add your result |
| MSE | Add your result |
| RMSE | Add your result |
| R² | Add your result |

## Screenshots

Add screenshots of the Streamlit app to the `screenshots/` folder, then embed them here:

```markdown
![Application home page](screenshots/home_page.png)
![Prediction result](screenshots/prediction_result.png)
```

## Important notes

- Keep `california_housing_model.pkl` in the same directory as `california_app.py`, unless you update the model path in the app.
- The app expects the model to accept the eight feature columns listed above, with the same names and order used in the training workflow.
- Scikit-learn model files are not guaranteed to be compatible across different library versions. For reliable loading, use the same Scikit-learn version that was used to train and save the model.
- Do not commit virtual environments, cache folders, or temporary editor files.

## Future improvements

- Add automated tests for input validation and predictions.
- Display model evaluation metrics in the app.
- Add feature-importance or model-explanation visualizations where supported.
- Deploy the Streamlit app so it can be accessed online.

## Author

**Hrishavh chandrawanshi**

[Add your GitHub profile link here.](https://github.com/hrishavh21)
