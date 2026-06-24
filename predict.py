import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from config import CONFIG


def forecast(excel_path=None):
    cfg = CONFIG
    if excel_path is None:
        excel_path = cfg["excel_path"]

    # Load excel
    df_raw = pd.read_excel(excel_path, sheet_name=cfg["sheet_name"], na_values=["NA","NaN","","null","?"])

    # Load artifacts
    pre_pipe = joblib.load(cfg["pipeline_path"])
    X_scaler = joblib.load(cfg["x_scaler_path"])
    y_scaler = joblib.load(cfg["y_scaler_path"])
    model = tf.keras.models.load_model(cfg["model_path"])

    # Preprocess
    df_prep = pre_pipe.transform(df_raw)

    feature_cols = cfg["feature_cols"]

    X_all = X_scaler.transform(df_prep[feature_cols]).astype("float32")

    H, HZ = cfg["history"], cfg["horizon"]
    if len(X_all) < H:
        raise ValueError(
            f"At least {H} preprocessed observations are required; "
            f"received {len(X_all)}."
        )

    # Use all observations through the latest timestamp as forecast history.
    X_last = X_all[-H:][np.newaxis, ...]
    pred_scaled = model.predict(X_last, verbose=0)[0] 
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()

    
    last_obs_ts = df_prep.index[-1]
    future_index = pd.date_range(
        start=last_obs_ts + pd.Timedelta(hours=1),
        periods=HZ, freq="H", tz="UTC"
    )

    return pd.Series(pred, index=future_index, name="RT_Demand_forecast")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    fc = forecast(path)
    print(fc.tail(30))
