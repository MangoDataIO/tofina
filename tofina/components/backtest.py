import torch
import pandas as pd
from typing import List, Union
import datetime as dt


def forecast(date: str, asset: str) -> torch.Tensor:
    pass


timeType = Union[
    str,
    pd.Timestamp,
    dt.datetime,
]


class Backtester:
    def __init__(self, timestamps: List[timeType], horizon=20):
        self.ts_dict = {}
        self.ticker_forecaster_map = {}
        self.timestamps = timestamps
        self.horizon = horizon

    def optionsDataFromDataFrame(self, df, ticker):
        def process_df(df: pd.DataFrame) -> pd.DataFrame:
            pass

        self.ts_dict[ticker] = process_df(df)

    def stockDataFromDataFrame(self, df, ticker):
        self.ts_dict[ticker] = df

    def registerForecaster(self, forecast, supported_tickers):
        for ticker in supported_tickers:
            self.ticker_forecaster_map[ticker] = forecast

    def backtest(self):
        pass
