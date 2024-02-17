import torch
import pandas as pd
from typing import List, Union, Dict
import datetime as dt
import tofina.components.portfolio as portfolio
import functools
from pathlib import Path
from tofina.components.instrument import NonDerivativePayout
from tofina.components.strategy import BuyAndHold
from tofina.macros.portfolioOptimization import optimizeStockPortfolioRiskAverse
from tqdm import tqdm


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
        self.timestamps = timestamps
        self.horizon = horizon
        self.pointInTimePortfolio: Dict[str, portfolio.Portfolio] = {}
        for timestamp in timestamps:
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

    def addStockToPortfolio(
        self,
        portfolio_: portfolio.Portfolio,
        ticker: str,
        price: float,
        allowShorting: bool = False,
        comission_dict: Dict[str, float] = {},
    ):
        if ticker in comission_dict:
            comission = comission_dict[ticker]
        else:
            comission = 0
        portfolio_.addInstrument(
            ticker,
            ticker + "_Stock",
            NonDerivativePayout,
            price,
            False,
            comission,
        )
        if allowShorting:
            portfolio_.addInstrument(
                ticker,
                ticker + "_Stock_Short",
                NonDerivativePayout,
                price,
                True,
                comission,
            )

    def registerForecaster(
        self,
        forecast: forecast,
        supported_tickers: List[str],
        allowShorting: bool = False,
        comission_dict: Dict[str, float] = {},
    ):
        for timestamp in self.timestamps:
            portfolio_ = self.pointInTimePortfolio[timestamp]
            for ticker in supported_tickers:
                portfolio_.addAsset(
                    ticker, functools.partial(forecast, date=timestamp, asset=ticker)
                )
                price = self.historicalTrajectory[ticker][timestamp][0]
                self.addStockToPortfolio(
                    portfolio_, ticker, price, allowShorting, comission_dict
                )

    def equal_weights(self, num_instruments: int):
        return [1 / num_instruments] * num_instruments

    def optimizeStrategy(self, logFolderPath: str):
        logFolderPath = Path(logFolderPath)
        for timestamp in tqdm(self.timestamps):
            portfolio_ = self.pointInTimePortfolio[timestamp]
            portfolio_.setStrategy(
                self.equal_weights(portfolio_.num_instruments), BuyAndHold
            )
            optimizeStockPortfolioRiskAverse(
                portfolio_,
                logFolderPath / f"portfolioOptimization_{timestamp}.csv",
            )

    def evaluateStrategy(self):
        pass

    def aggregateResults(self):
        pass
