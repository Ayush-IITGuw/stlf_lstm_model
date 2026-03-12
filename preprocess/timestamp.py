import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimestampBuilder(BaseEstimator, TransformerMixin):
 def __init__(self,tz_local="America/New_York",date_col="Date",hour_col="Hr_End",target_col="RT_Demand",drop_cols=None):
	self.tz_local = tz_local
	self.date_col = date_col
	self.hour_col = hour_col
	self.target_col = target_col
	self.drop_cols = drop_cols or []

 def fit(self, X,y=None):
    return self

 def transform(self, X):
    """
    Build UTC timestamp index, fixing '24:00' rollover and DST ambiguous hours.
    Expects columns: 'Date' (YYYY-MM-DD), 'Hr_End' (e.g. '00','01',...,'24' or '02X' for repeat).
    """
    df = X.copy()

    # Identify original 24:00 rows 
    mask_24 = df["Hr_End"].astype(str).eq("24")
    df.loc[mask_24, "Hr_End"] = "00"
    df.loc[mask_24, "Date"] = pd.to_datetime(df.loc[mask_24, "Date"]) + pd.Timedelta(days=1)

    # Mark repeated hour via 'X' (Day Light Saving)
    df["dst_repeat"] = df["Hr_End"].astype(str).str.contains("X")
    df["hour_clean"] = df["Hr_End"].astype(str).str.replace("X", "")

    # Build naive timestamp, then localize and convert to UTC
    df["timestamp_naive"] = pd.to_datetime(df["Date"].astype(str) + " " + df["hour_clean"])
    df["timestamp_local"] = df["timestamp_naive"].dt.tz_localize(
        self.tz_local,
        ambiguous=~df["dst_repeat"]  
    )
    df["timestamp_utc"] = df["timestamp_local"].dt.tz_convert("UTC")

    # Final index
    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp").sort_index()

    df.drop(columns=["timestamp_naive","timestamp_local","timestamp_utc","hour_clean","dst_repeat"], inplace =  True)
    return df
