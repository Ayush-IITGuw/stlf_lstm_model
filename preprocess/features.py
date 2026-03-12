from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureMaker(BaseEstimator, TransformerMixin):
 
 def __init__(self, target_col="RT_Demand", drop_cols=None):
        self.target_col = target_col
        self.drop_cols = drop_cols or []

 
 def fit(self, X, y=None):
        return self

 def transform(self, X):
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["demand_lag_1"] = df[target].shift(1)
    df["demand_lag_3"] = df[target].shift(3)
    df["demand_lag_24"] = df[target].shift(24)
    df["rolling_mean"] = df[target].rolling(window=48, min_periods=1).mean()
    df = relevant_select(df,drop_cols)
    return na_clean(df)


def relevant_select(df, drop_cols):
    df = df.copy()
    dropper = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=dropper)


def na_clean(df):
    return df.dropna()
