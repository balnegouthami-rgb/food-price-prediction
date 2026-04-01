Alright — here’s your fully updated, clean, production-style React + FastAPI project with:

✅ Default dataset loading
✅ Model selection from UI
✅ Automatic .pkl saving
✅ Option to use saved model (no retraining)
✅ Proper folder structure & paths
✅ Clean UI + backend

🧠 📁 FULL PROJECT STRUCTURE
project/
│
├── backend/
│   ├── main.py
│   ├── utils.py
│   ├── data/
│   │     └── merged_final.csv
│   ├── models/
│   │     ├── rf_model.pkl
│   │     ├── xgb_model.pkl
│   │     └── lgb_model.pkl
│
├── frontend/
│   ├── src/
│   │     └── App.js
│   └── package.json
⚙️ BACKEND (FastAPI)
🔧 Install
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm joblib
📄 backend/utils.py
import pandas as pd

def preprocess(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    cols_to_drop = ['name', 'stations', 'description', 'icon', 'sunrise', 'sunset', 'severerisk']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    df = df.sort_values('date')

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].interpolate(method='linear').fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df = pd.get_dummies(df, drop_first=True)
    df = df.drop_duplicates()

    return df
🚀 backend/main.py
from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import preprocess

app = FastAPI()

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "merged_final.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# MODEL GETTER
# ===============================
def get_model(model_name):
    if model_name == "rf":
        return RandomForestRegressor()
    elif model_name == "xgb":
        return XGBRegressor()
    elif model_name == "lgb":
        return LGBMRegressor()
    else:
        return None

# ===============================
# MAIN API
# ===============================
@app.get("/run-model")
def run_model(model_name: str = "rf", use_saved: bool = False):

    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    train = df[df['year'] < 2019]
    test = df[df['year'] == 2019]

    X_train = train.drop(['value', 'date', 'year'], axis=1)
    y_train = train['value']
    X_test = test.drop(['value', 'date', 'year'], axis=1)
    y_test = test['value']

    X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")

    # ===============================
    # LOAD OR TRAIN MODEL
    # ===============================
    if use_saved and os.path.exists(model_path):
        model = joblib.load(model_path)
        status = "Loaded saved model"
    else:
        model = get_model(model_name)
        if model is None:
            return {"error": "Invalid model"}

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        status = "Model trained and saved"

    # ===============================
    # PREDICTION
    # ===============================
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {
        "status": status,
        "model": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "predictions": preds[:20].tolist()
    }
▶️ Run Backend
cd backend
uvicorn main:app --reload
🎨 FRONTEND (React)
🔧 Install
npx create-react-app frontend
cd frontend
npm install axios
💻 frontend/src/App.js
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [model, setModel] = useState("rf");
  const [useSaved, setUseSaved] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runModel = async () => {
    setLoading(true);
    try {
      const res = await axios.get(
        `http://127.0.0.1:8000/run-model?model_name=${model}&use_saved=${useSaved}`
      );
      setResult(res.data);
    } catch (err) {
      alert("Error connecting to backend");
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>📊 Commodity Price Prediction</h1>

      {/* MODEL SELECT */}
      <div>
        <label>Select Model: </label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="rf">Random Forest</option>
          <option value="xgb">XGBoost</option>
          <option value="lgb">LightGBM</option>
        </select>
      </div>

      <br />

      {/* USE SAVED MODEL */}
      <div>
        <label>
          <input
            type="checkbox"
            checked={useSaved}
            onChange={() => setUseSaved(!useSaved)}
          />
          Use Saved Model (Skip Training)
        </label>
      </div>

      <br />

      <button onClick={runModel}>
        {loading ? "Running..." : "🚀 Run Model"}
      </button>

      {/* RESULTS */}
      {result && (
        <div style={{ marginTop: "30px" }}>
          <h2>Results</h2>

          <p><b>Status:</b> {result.status}</p>
          <p><b>Model:</b> {result.model}</p>
          <p><b>MAE:</b> {result.mae}</p>
          <p><b>RMSE:</b> {result.rmse}</p>

          <h3>Sample Predictions</h3>
          <ul>
            {result.predictions.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
▶️ Run Frontend
cd frontend
npm start
🔥 FINAL FEATURES
🎯 Your app now does:
✅ Loads dataset automatically
✅ Select model from UI
✅ Train OR reuse saved model
✅ Saves .pkl automatically
✅ Displays metrics + predictions
⚡ PRODUCTION IMPROVEMENTS (Next Step)

If you want this to look like a real product, I can upgrade:

🎨 UI
Tailwind CSS (modern UI)
Charts (Recharts)
Cards + dashboard
🧠 ML
Optuna tuning in backend
Auto best model selection
🌐 Deploy
Backend → Render
Frontend → Vercel