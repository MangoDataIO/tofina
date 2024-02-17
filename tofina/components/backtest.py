import torch
import pandas as pd
from typing import List, Union
import datetime as dt
import tofina.components.portfolio as portfolio


def forecast(date: str, asset: str) -> torch.Tensor:
    pass


timeType = Union[
    str,
    pd.Timestamp,
    dt.datetime,
]


class Backtester:
    def __init__(self, timestamps: List[timeType], horizon=20, monteCarloTrials=1000):
        self.historicalTrajectory = {}
        self.ticker_forecaster_map = {}
        self.timestamps = timestamps
        self.horizon = horizon
        self.pointInTimePortfolio = {}
        for timestamp in timestamp:
            self.pointInTimePortfolio[timestamp] = portfolio.Portfolio(
                processLength=horizon, monteCarloTrials=monteCarloTrials
            )

    def stockDataFromDataFrame(self, df: pd.DataFrame, ticker: str):
        self.historicalTrajectory[ticker] = {}
        for timestamp in self.timestamps:
            index = df.index.get_loc(timestamp)
            self.historicalTrajectory[ticker][timestamp] = df["Close"].values[
                index : index + self.horizon
            ]

    def registerForecaster(self, forecast, supported_tickers):
        for ticker in supported_tickers:
            self.ticker_forecaster_map[ticker] = forecast

    def backtest(self):
        pass
