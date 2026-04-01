from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# CORS setup to allow React frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_final.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    df = df.drop(columns=['name','stations','description','icon','sunrise','sunset','severerisk'], errors='ignore')

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    df[num_cols] = df[num_cols].interpolate().fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df['year'] = df['date'].dt.year

    df = pd.get_dummies(df, drop_first=True)
    df = df.drop_duplicates()

    return df

def rf_objective(trial, X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 120),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def xgb_objective(trial, X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 120),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        random_state=42,
        verbosity=0,
        eval_metric='rmse'
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

def lgb_objective(trial, X_train, y_train, X_val, y_val):
    model = LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 120),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

@app.get("/run-model")
def run_model(
    model_name: str = "rf",
    train_end_year: int = 2018,
    test_start_date: str = "2019-01-01",
    test_end_date: str = "2019-12-31",
    use_saved: bool = False
):
    try:
        print("\n🚀 API HIT")

        train_end_year = int(train_end_year)
        test_start_date = pd.to_datetime(test_start_date)
        test_end_date = pd.to_datetime(test_end_date)

        df = pd.read_csv(DATA_PATH)
        df = preprocess(df)

        if 'value' not in df.columns:
            return {"error": "'value' column missing"}

        train = df[df['year'] <= train_end_year]
        test = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]

        if train.empty or test.empty:
            return {"error": "Invalid date range"}

        # Clean column names
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

        X_train = train.drop(['value','date','year'], axis=1)
        y_train = train['value']
        X_test = test.drop(['value','date','year'], axis=1)
        y_test = test['value']

        # Clean again (safe)
        X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

        # Align features
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

        if use_saved and os.path.exists(model_path):
            model = joblib.load(model_path)
            status = "Loaded saved model"
        else:
            study = optuna.create_study(direction='minimize')

            if model_name == "rf":
                study.optimize(lambda t: rf_objective(t,X_train,y_train,X_test,y_test), n_trials=5)
                model = RandomForestRegressor(**study.best_params, random_state=42)

            elif model_name == "xgb":
                study.optimize(lambda t: xgb_objective(t,X_train,y_train,X_test,y_test), n_trials=5)
                model = XGBRegressor(**study.best_params, random_state=42, verbosity=0, eval_metric='rmse')

            elif model_name == "lgb":
                study.optimize(lambda t: lgb_objective(t,X_train,y_train,X_test,y_test), n_trials=5)
                model = LGBMRegressor(**study.best_params, random_state=42)

            else:
                return {"error": "Invalid model"}

            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            status = "Model trained"

        preds = model.predict(X_test)
        errors = (y_test - preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Limit to first 100 for frontend performance
        n = min(100, len(preds))

        return {
            "status": status,
            "model": model_name,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "predictions": preds[:n].tolist(),
            "actual": y_test[:n].tolist(),
            "dates": test['date'].astype(str).iloc[:n].tolist(),
            "errors": errors[:n].tolist()
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}