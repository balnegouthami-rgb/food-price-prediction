import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("D:\\uday\\food commodity price prediction\\merged_final.csv")

# ===============================
# 2. CONVERT DATE
# ===============================
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ===============================
# 3. DROP UNNECESSARY COLUMNS
# ===============================
cols_to_drop = ['name', 'stations', 'description', 'icon', 'sunrise', 'sunset', 'severerisk']
df = df.drop(columns=cols_to_drop, errors='ignore')

# ===============================
# 4. SORT DATA
# ===============================
df = df.sort_values('date')

# ===============================
# 5. HANDLE MISSING VALUES
# ===============================
# Numerical → interpolate then fill remaining with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].interpolate(method='linear')
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# ===============================
# 6. FEATURE ENGINEERING
# ===============================
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# ===============================
# 7. ENCODE CATEGORICAL DATA
# ===============================
df = pd.get_dummies(df, drop_first=True)

# ===============================
# 8. REMOVE DUPLICATES
# ===============================
df = df.drop_duplicates()

# ===============================
# 9. TRAIN-TEST SPLIT (2014–2018 train, 2019 test)
# ===============================
train = df[df['year'] < 2019]
test = df[df['year'] == 2019]

# ===============================
# 10. FEATURES & TARGET
# ===============================
X_train = train.drop(['value', 'date', 'year'], axis=1)
y_train = train['value']

X_test = test.drop(['value', 'date', 'year'], axis=1)
y_test = test['value']

# ===============================
# 11. CLEAN FEATURE NAMES FOR ALL MODELS
# ===============================
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# ===============================
# 12. RANDOM FOREST
# ===============================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest MAE:", mae_rf)
print("Random Forest RMSE:", rmse_rf)

# ===============================
# 13. XGBOOST
# ===============================
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("XGBoost MAE:", mae_xgb)
print("XGBoost RMSE:", rmse_xgb)

# ===============================
# 14. LIGHTGBM
# ===============================
lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

print("LightGBM MAE:", mae_lgb)
print("LightGBM RMSE:", rmse_lgb)
# ===============================
# 15. FINAL DATA CHECK
# ===============================
print(df.isnull().sum())
print(df.head())  // hyperparameter tuning chesi and graphs and ui cheyali