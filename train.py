import os
import joblib
import numpy as np
import pandas as pd

from config import CONFIG
from preprocess.timestamp import TimestampBuilder
from preprocess.features import FeatureMaker
from preprocess.scalers import scale_data
from preprocess.window import make_windows
from model.lstm import build_lstm
from model.metrics import regression_report

from sklearn.pipeline import Pipeline

def main():

	cfg = CONFIG
	os.makedirs(cfg["artifacts_dir"], exist_ok=True)

#1 Load Data
	df = pd.read_excel(cfg["excel_path"], sheet_name=cfg["sheet_name"], na_values=["NA","NaN","","null","?"])

#2 Preprocessing
	pre_pipe = Pipeline(steps=[
        ("timestamps", TimestampBuilder(tz_local=cfg["tz_local"])),
        ("features", FeatureMaker(target_col=cfg["target_col"], drop_cols=cfg["drop_cols"]))
        ])

	df_prep = pre_pipe.fit_transform(df)

#3 Train/Test Split
	cutoff = pd.Timestamp(cfg["train_cutoff_utc"])
	train = df_prep.loc[:cutoff].copy()
	test  = df_prep.loc[cutoff + pd.Timedelta(seconds=1):].copy()

	feature_cols = cfg["feature_cols"]
	ycol = cfg["target_col"]

#4 Scale Data
    X_scaler = make_scaler(cfg["x_scaler"]).fit(train[feature_cols])
    y_scaler = make_scaler(cfg["y_scaler"]).fit(train[[ycol]])

    X_train = X_scaler.transform(train[feature_cols]).astype("float32")
    y_train = y_scaler.transform(train[[ycol]]).astype("float32").ravel()
    X_test  = X_scaler.transform(test[feature_cols]).astype("float32")
    y_test  = y_scaler.transform(test[[ycol]]).astype("float32").ravel()

#5 Make Windows
    HISTORY, HORIZON = cfg["history"], cfg["horizon"]
    X_tr_w, y_tr_w = make_windows(X_train, y_train, HISTORY, HORIZON)
    X_te_w, y_te_w = make_windows(X_test,  y_test,  HISTORY, HORIZON)

    n_features = X_tr_w.shape[-1]

#6 Train Model
	model = build_lstm(HISTORY, n_features, HORIZON)
	es = tf.keras.callbacks.EarlyStopping(
        	monitor="val_loss", patience=cfg["patience"], restore_best_weights=True
	)


	history = model.fit(
        	X_tr_w, y_tr_w,
	        validation_split=cfg["validation_split"],
	        shuffle=cfg["shuffle"],
	        epochs=cfg["epochs"],
	        batch_size=cfg["batch_size"],
        	callbacks=[es],
	        verbose=1
	)


#7 Evaluate
	y_pred_scaled = model.predict(X_te_w)
	y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape((-1,1)).reshape(-1,HORIZON))
	y_true = y_scaler.inverse_transform(y_te_w.reshape(-1,1)).reshape(-1, HORIZON)

	report = regression_report(y_true, y_pred)
	print(report)

#8 Preserve Model
    joblib.dump(pre_pipe, cfg["pipeline_path"])
    joblib.dump(X_scaler, cfg["x_scaler_path"])
    joblib.dump(y_scaler, cfg["y_scaler_path"])
    model.save(cfg["model_path"])

if __name__ == "__main__":
	main()
