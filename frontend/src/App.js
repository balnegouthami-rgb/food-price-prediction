import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  Button,
  TextField,
  FormControlLabel,
  Checkbox,
  Tabs,
  Tab,
  Box,
  Paper,
  CircularProgress,
  Alert,
} from "@mui/material";
import { Line, Bar } from "react-chartjs-2";
import "chart.js/auto";

const BACKEND_URL = "http://localhost:8000";

function App() {
  const [tab, setTab] = useState("rf");
  const [trainYear, setTrainYear] = useState(2018);
  const [testStartDate, setTestStartDate] = useState("2019-01-01");
  const [testEndDate, setTestEndDate] = useState("2019-12-31");
  const [useSaved, setUseSaved] = useState(false);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({});
  const [errorMsg, setErrorMsg] = useState("");

  const handleRunModel = async (model) => {
    setErrorMsg("");
    setLoading(true);
    try {
      const res = await axios.get(`${BACKEND_URL}/run-model`, {
        params: {
          model_name: model,
          train_end_year: trainYear,
          test_start_date: testStartDate,
          test_end_date: testEndDate,
          use_saved: useSaved,
        },
      });

      if (res.data.error) {
        setErrorMsg(res.data.error);
        setResults({});
      } else {
        setResults((prev) => ({ ...prev, [model]: res.data }));
        setTab(model);
      }
    } catch (e) {
      setErrorMsg("Failed to fetch model results.");
      setResults({});
    }
    setLoading(false);
  };

  const handleRunAll = () => {
    // Run all 3 models one by one (async)
    ["rf", "xgb", "lgb"].forEach((model) => handleRunModel(model));
  };

  const renderResults = (model) => {
    const data = results[model];
    if (!data) return null;

    // Prepare Actual vs Predicted line chart
    const lineData = {
      labels: data.dates,
      datasets: [
        {
          label: "Actual",
          data: data.actual,
          borderColor: "green",
          backgroundColor: "green",
          tension: 0.3,
        },
        {
          label: "Predicted",
          data: data.predictions,
          borderColor: "red",
          backgroundColor: "red",
          tension: 0.3,
        },
      ],
    };

    // Prepare Error Distribution histogram (bar chart)
    const bins = {};
    data.errors.forEach((e) => {
      const key = Math.round(e);
      bins[key] = (bins[key] || 0) + 1;
    });

    const errorDistData = {
      labels: Object.keys(bins),
      datasets: [
        {
          label: "Count",
          data: Object.values(bins),
          backgroundColor: "rgba(255,99,132,0.6)",
        },
      ],
    };

    return (
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          {model.toUpperCase()} Results
        </Typography>

        <Typography>
          <b>Status:</b> {data.status}
        </Typography>
        <Typography>
          <b>MAE:</b> {data.mae}
        </Typography>
        <Typography>
          <b>RMSE:</b> {data.rmse}
        </Typography>

        <Box mt={3}>
          <Typography variant="h6">Actual vs Predicted</Typography>
          <Line data={lineData} />
        </Box>

        <Box mt={3}>
          <Typography variant="h6">Error Distribution</Typography>
          <Bar data={errorDistData} />
        </Box>
      </Box>
    );
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box display="flex" alignItems="center" mb={2}>
        <img
          src="https://cdn-icons-png.flaticon.com/512/4144/4144413.png"
          alt="commodity"
          width={40}
          height={40}
          style={{ marginRight: 8 }}
        />
        <Typography variant="h4" fontWeight="bold" color="green">
          Commodity Price Prediction (2014-2019)
        </Typography>
      </Box>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" gap={2} flexWrap="wrap" alignItems="center">
          <TextField
            label="Train End Year"
            type="number"
            value={trainYear}
            onChange={(e) => setTrainYear(e.target.value)}
            sx={{ width: 140 }}
            inputProps={{ min: 2014, max: 2028 }}
          />
          <TextField
            label="Test Start Date"
            type="date"
            value={testStartDate}
            onChange={(e) => setTestStartDate(e.target.value)}
            sx={{ width: 180 }}
            InputLabelProps={{ shrink: true }}
            inputProps={{
              min: "2014-01-01",
              max: "2029-12-31",
            }}
          />
          <TextField
            label="Test End Date"
            type="date"
            value={testEndDate}
            onChange={(e) => setTestEndDate(e.target.value)}
            sx={{ width: 180 }}
            InputLabelProps={{ shrink: true }}
            inputProps={{
              min: "2014-01-01",
              max: "2029-12-31",
            }}
          />
        </Box>

        <FormControlLabel
          control={
            <Checkbox
              checked={useSaved}
              onChange={(e) => setUseSaved(e.target.checked)}
            />
          }
          label="Use Saved Model (Skip Training)"
          sx={{ mt: 1 }}
        />

        <Box mt={2} display="flex" gap={2} flexWrap="wrap">
          {["rf", "xgb", "lgb"].map((m) => (
            <Button
              key={m}
              variant={tab === m ? "contained" : "outlined"}
              color="success"
              onClick={() => handleRunModel(m)}
              sx={{ minWidth: 120 }}
            >
              {m === "rf"
                ? "Random Forest"
                : m === "xgb"
                ? "XGBoost"
                : "LightGBM"}
            </Button>
          ))}
        </Box>

        <Box mt={3}>
          <Button
            variant="contained"
            color="success"
            onClick={handleRunAll}
            sx={{ minWidth: 200, fontWeight: "bold" }}
          >
            🚀 Run All Models
          </Button>
        </Box>
      </Paper>

      {loading && (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      )}

      {errorMsg && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {errorMsg}
        </Alert>
      )}

      {renderResults(tab)}
    </Container>
  );
}

export default App;