# Short-Term Load Forecasting with Attention-Based LSTM: ISO-NE Maine Demand Forecasting

A quantitative forecasting project that predicts near-term electricity demand using an
encoder-decoder LSTM with Bahdanau attention over hourly ISO New England load, market, and
weather data.

## Overview

Short-term load forecasting is the task of estimating electricity demand over the next few hours
or days from recent demand behavior, weather conditions, and calendar patterns. This project
implements an end-to-end forecasting pipeline for **Maine (`ME`) real-time demand (`RT_Demand`)**
using hourly records from `data/load_data.xlsx`:

1. **Timestamp normalization** - converts hour-ending ISO-NE records into a UTC timestamp index,
   including `24:00` rollover handling and repeated daylight-saving hours.
2. **Feature engineering** - creates lagged demand features, a rolling demand mean, weather
   variables, and a cyclical day-of-week feature.
3. **Sequence construction** - converts the tabular time series into supervised windows with a
   168-hour lookback and a 24-hour forecast horizon.
4. **Deep learning forecast** - trains a PyTorch encoder-decoder LSTM with Bahdanau attention,
   using teacher forcing during training and autoregressive decoding during evaluation.
5. **Evaluation** - inverse-transforms predictions back to demand scale and reports MAE, RMSE,
   and R2 across the forecast horizon.

## Architecture

```text
Data Layer          - Excel workbook (`data/load_data.xlsx`), sheet-level ISO-NE regional data
Preprocessing Layer - TimestampBuilder: local hour-ending records -> UTC datetime index
Feature Layer       - FeatureMaker: demand lags, 48-hour rolling mean, weather, day-of-week signal
Windowing Layer     - 168-hour historical windows -> 24-hour multi-step targets
Model Layer         - PowerEncoder + Bahdanau attention + PowerDecoder seq2seq LSTM
Evaluation Layer    - Inverse scaling, MAE, RMSE, R2, actual-vs-predicted visualization
Artifact Layer      - Best PyTorch checkpoint plus preprocessing and scaler objects
```

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch (`nn.LSTM`, encoder-decoder sequence model, attention) |
| Machine Learning Utilities | scikit-learn (`Pipeline`, `StandardScaler`, `MinMaxScaler`, metrics) |
| Data Handling | `pandas`, `numpy`, Excel input via `read_excel` |
| Serialization | `joblib`, `torch.save` |
| Visualization | `matplotlib` |

## Repository Structure

```text
train_new.py              Main training, validation, checkpointing, and evaluation script
config.py                 Central configuration for data paths, features, split, and model params
data/load_data.xlsx       ISO-NE workbook with regional demand, price, and weather sheets
preprocess/timestamp.py   TimestampBuilder transformer for hour-ending and DST handling
preprocess/features.py    FeatureMaker transformer for lags, rolling mean, and calendar features
preprocess/scalers.py     Scaler factory for standard and min-max scaling
preprocess/window.py      Sliding-window sequence builder
model/PowerModel.py       Encoder, Bahdanau attention, and decoder modules
model/metrics.py          MAE, RMSE, and R2 reporting
model/evaluator.py        Small evaluator wrapper around model_report
```

## Methodology

### 1. Data source

The workbook contains ISO New England demand and market data across multiple regional sheets,
including `ISO NE CA`, `ME`, `NH`, `VT`, `CT`, `RI`, `SEMA`, `WCMA`, and `NEMA`. The current
configuration trains on the `ME` sheet and targets `RT_Demand`.

The `ME` sheet contains hourly observations from **2024-01-01** through **2025-12-31** with
columns for day-ahead demand, real-time demand, locational marginal price components, dry bulb
temperature, and dew point.

### 2. Time handling

`TimestampBuilder` turns `Date` and `Hr_End` into a timezone-aware UTC index. It handles:

- `Hr_End == 24` by rolling the timestamp into `00:00` on the next date.
- repeated daylight-saving hours marked with `X`.
- local timezone localization with `America/New_York`.
- final conversion to UTC for consistent time-series splitting.

### 3. Feature engineering

`FeatureMaker` builds the model inputs from the target history, weather, and calendar structure:

| Feature | Description |
|---|---|
| `demand_lag_1` | previous-hour real-time demand |
| `demand_lag_3` | demand three hours earlier |
| `demand_lag_24` | demand at the same hour on the previous day |
| `demand_lag_168` | demand at the same hour one week earlier |
| `rolling_mean` | 48-hour rolling mean of `RT_Demand` |
| `Dry_Bulb` | dry bulb temperature |
| `Dew_Point` | dew point temperature |
| `dow_sin` | sine-encoded day-of-week feature |

Columns not used for forecasting, including price components and day-ahead demand, are dropped in
`config.py`.

### 4. Train/test split and scaling

The configured split uses:

```text
Train cutoff: 2025-10-01 00:00:00+00:00
Input scaler: StandardScaler
Target scaler: MinMaxScaler
History window: 168 hours
Forecast horizon: 24 hours
Batch size: 64
Hidden size: 32
Epochs: 40
Learning rate: 0.003
Dropout: 0.3
```

Scalers are fitted only on the training partition, then reused for the test partition.

### 5. Forecasting model

The model is implemented in `model/PowerModel.py`:

- `PowerEncoder` projects input features into the hidden dimension and encodes the 168-hour
  history with an LSTM.
- `PowerAttention` applies Bahdanau attention over encoder outputs at each forecast step.
- `PowerDecoder` predicts the 24-hour horizon one step at a time, conditioning on attention
  context and either the true previous target during teacher-forced training or its own prior
  prediction during evaluation.

Training uses mean squared error loss and Adam optimization. The best validation checkpoint is
saved to `artifacts/best_power_seq2seq.pt`.

### 6. Evaluation

Predictions and targets are inverse-transformed from scaled space back to the original demand
scale before metrics are computed. The evaluation flattens the 24-step horizons and reports:

- MAE: mean absolute error
- RMSE: root mean squared error
- R2: coefficient of determination

The training script also plots the first 200 actual vs predicted test-horizon values.

## Results

| Metric | Value |
|---|---:|
| Mean Absolute Error | 69.65 |
| Root Mean Squared Error | 97.70 |
| R2 Score | 0.8157 |

The R2 value indicates that the model explains about 80% of the variance in the evaluated
real-time demand targets. MAE and RMSE are reported in the original `RT_Demand` scale after
inverse transformation.

## Getting Started

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a local run from the repository root, update `CONFIG["excel_path"]` in `config.py` from the
Colab-style path:

```python
"/content/stlf/data/load_data.xlsx"
```

to:

```python
"data/load_data.xlsx"
```

Then run:

```bash
python train_new.py
```

The script reads the Excel data, builds preprocessing features, trains the attention-based
sequence model, evaluates the best checkpoint, displays an actual-vs-predicted plot, and writes
artifacts to `artifacts/`.

## Outputs

```text
artifacts/best_power_seq2seq.pt      Best encoder/decoder checkpoint
artifacts/preprocess_pipeline.pkl    Fitted preprocessing pipeline
artifacts/x_scaler.pkl               Fitted feature scaler
artifacts/y_scaler.pkl               Fitted target scaler
```

## Future Work

- Compare against statistical baselines such as seasonal naive, SARIMA, and gradient boosting.
- Add hour-of-day and holiday features for stronger calendar representation.

## License

MIT
