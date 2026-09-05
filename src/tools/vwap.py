"""VWAP and Volume Profile calculations."""
import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


def compute_volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    price_min = df["low"].min()
    price_max = df["high"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_at_price = np.zeros(bins)

    for _, row in df.iterrows():
        for i in range(bins):
            if row["low"] <= edges[i + 1] and row["high"] >= edges[i]:
                vol_at_price[i] += row["volume"]

    return pd.DataFrame({"price": centers, "volume": vol_at_price})
