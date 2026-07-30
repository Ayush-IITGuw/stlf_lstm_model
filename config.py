from datetime import datetime,timezone

CONFIG = {
    # Data
    "excel_path": "/content/stlf/data/load_data.xlsx",
    "sheet_name": "ME",

    # Time handling
    "tz_local": "America/New_York",
    "train_cutoff_utc": "2025-10-01 00:00:00+00:00",

    # Columns
    "target_col": "RT_Demand",
    "feature_cols": [
        "demand_lag_1", "demand_lag_3", "demand_lag_24", 
        "Dry_Bulb", "Dew_Point", "rolling_mean", "dow_sin", "demand_lag_168"
    ],
    "drop_cols": ["DA_LMP","DA_EC","DA_CC","DA_MLC","RT_LMP",
                  "RT_EC","RT_CC","RT_MLC","Date","Hr_End", "dayofweek", "DA_Demand",],

    # Sequence settings
    "history": 168,
    "horizon": 24,

    # Training
    "batch_size": 64,
    "hidden_size":32,
    "epochs": 30,
    "patience": 6,
    "validation_split": 0.1,
    "shuffle": False,
    "lr" : 0.003,
    "dropout" : 0.3,

    # Scaling
    "x_scaler": "standard",  
    "y_scaler": "minmax",  

    # Artifacts
    "artifacts_dir": "artifacts",
    "pipeline_path": "artifacts/preprocess_pipeline.pkl",
    "x_scaler_path": "artifacts/x_scaler.pkl",
    "y_scaler_path": "artifacts/y_scaler.pkl",
    "model_path": "artifacts/stlf_lstm.keras",
}
