import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
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
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].interpolate(method='linear').fillna(df[num_cols].median())

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
# 9. TRAIN-TEST SPLIT
# ===============================
train = df[df['year'] < 2019]
test = df[df['year'] == 2019]

X_train = train.drop(['value', 'date', 'year'], axis=1)
y_train = train['value']
X_test = test.drop(['value', 'date', 'year'], axis=1)
y_test = test['value']

# ===============================
# 10. CLEAN FEATURE NAMES
# ===============================
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# ===============================
# 11. OPTUNA HYPERPARAMETER TUNING
# ===============================
tscv = TimeSeriesSplit(n_splits=5)

# ---------- RANDOM FOREST ----------
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': 42
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    return np.mean(scores)

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(rf_objective, n_trials=10)

# ---------- XGBOOST ----------
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=10)

# ---------- LIGHTGBM ----------
def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(lgb_objective, n_trials=10)

# ===============================
# 12. TRAIN FINAL MODELS WITH BEST PARAMETERS
# ===============================
# Random Forest
rf_model = RandomForestRegressor(**study_rf.best_params)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# XGBoost
xgb_model = XGBRegressor(**study_xgb.best_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

# LightGBM
lgb_model = LGBMRegressor(**study_lgb.best_params)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM MAE:", mean_absolute_error(y_test, y_pred_lgb))
print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lgb)))

# ===============================
# 13. FINAL DATA CHECK
# ===============================
print(df.isnull().sum())
print(df.head())