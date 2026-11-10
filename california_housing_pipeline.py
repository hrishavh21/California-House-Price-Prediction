import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset and prepare DataFrame
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Check structure and data hygiene
print(df.head())
print(df.shape)
print(df.info())

# Checking for null values and duplicate values
print(df.isnull().sum())
print(df.duplicated().sum())

# Feature correlation with target
print(df.corr(numeric_only=True)['MedHouseVal'].sort_values(ascending=False))
# 1. Histograms to check skewness
df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig('outputs/1_histograms_skewness.png', dpi=300)
plt.close()

# 2. Box plots for distribution and quartiles
df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8))
plt.tight_layout()
plt.savefig('outputs/2_boxplots_distribution.png', dpi=300)
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.tight_layout()
plt.savefig('outputs/3_correlation_heatmap.png', dpi=300)
plt.close()

# 4. Scatter plot of top correlated feature vs target
plt.figure(figsize=(7, 5))
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3)
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.tight_layout()
plt.savefig('outputs/4_scatter_medinc_vs_target.png', dpi=300)
plt.close()
# Check and remove capped values (values >= 5.0) in MedHouseVal
print(df['MedHouseVal'].max())
print((df['MedHouseVal'] >= 5.0).sum())

df = df[df['MedHouseVal'] < 5.0]  # Capped off the values = 5.0
print(df.shape)

# Outlier filtering using IQR
cols = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
print(df[cols].describe())

for col in cols:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  df = df[(df[col] >= lower) & (df[col] <= upper)]

print(df[cols].describe())
print(df.shape)
# Split features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# 2. Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 3. HistGradientBoosting Regressor
hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_train, y_train)
hgb_pred = hgb.predict(X_test)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [
        mean_squared_error(y_test, lr_pred),
        mean_squared_error(y_test, rf_pred),
        mean_squared_error(y_test, hgb_pred),
    ],
    'MAE': [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, hgb_pred),
    ],
    'R2 Score': [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, hgb_pred),
    ],
})

print(results)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [200, 400, 600],
    'max_leaf_nodes': [15, 31, 63],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [20, 40],
    'l2_regularization': [0, 0.1, 1.0],
}

gb_search = RandomizedSearchCV(
    estimator=HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

gb_search.fit(X_train, y_train)
y_pred_gb = gb_search.predict(X_test)

print('Best params:', gb_search.best_params_)
print('Best score:', gb_search.best_score_)

# Evaluation Metrics
print('MAE:', mean_absolute_error(y_test, y_pred_gb))
print('MSE:', mean_squared_error(y_test, y_pred_gb))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_gb)))
print('R2:', r2_score(y_test, y_pred_gb))

# Actual vs Predicted Plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_gb, alpha=0.3)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Gradient Boosting: Actual vs Predicted')
plt.savefig('outputs/actual_vs_predicted.png', dpi=300)
import joblib

# Export and reload model
joblib.dump(gb_search, 'california_housing_model.pkl')
loaded_model = joblib.load('california_housing_model.pkl')

# Mock test sample
new_house = pd.DataFrame({
    'MedInc': [5.0],
    'HouseAge': [20],
    'AveRooms': [5.5],
    'AveBedrms': [1.0],
    'Population': [1200],
    'AveOccup': [3.0],
    'Latitude': [34.05],
    'Longitude': [-118.25],
})

pred = loaded_model.predict(new_house)[0]
print(f'Predicted price: ${round(pred * 100000)}')