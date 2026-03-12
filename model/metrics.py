from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def model_report(y_true, y_predict):
	
    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    rmse = mean_squared_error(y_true.ravel(), y_pred.ravel(), squared=False) 
    r2 = r2_score(y_true.ravel(), y_pred.ravel())
    return {"mae": mae, "rmse": rmse, "r2": r2}

