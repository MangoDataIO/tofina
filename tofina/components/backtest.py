import torch
import pandas as pd
from typing import List, Union, Dict
import datetime as dt
import tofina.components.portfolio as portfolio
import functools
from pathlib import Path
from tofina.components.instrument import NonDerivativePayout, NonDerivativePayoutShort
from tofina.components.strategy import BuyAndHold
from tofina.macros.portfolioOptimization import optimizeStockPortfolioRiskAverse
from tofina.components.optimizer import Optimizer
from tqdm import tqdm
from tofina.utils import softmaxInverse


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
        self.pointInTimePortfolio: Dict[timeType, portfolio.Portfolio] = {}
        self.optimizers: Dict[timeType, Optimizer] = {}
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
            comission,
        )
        if allowShorting:
            portfolio_.addInstrument(
                ticker,
                ticker + "_Stock_Short",
                NonDerivativePayoutShort,
                price,
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

                def processFn(processLength, monteCarloTrials, params):
                    return forecast(
                        timestamp, ticker, processLength, monteCarloTrials, **params
                    )

                portfolio_.addAsset(ticker, processFn)
                price = self.historicalTrajectory[ticker][timestamp][0]
                self.addStockToPortfolio(
                    portfolio_, ticker, price, allowShorting, comission_dict
                )

    def optimizeStrategy(self, logFolderPath: str):
        logFolderPath = Path(logFolderPath)
        for timestamp in tqdm(self.timestamps):
            portfolio_ = self.pointInTimePortfolio[timestamp]
            portfolio_.setStrategy(
                torch.rand(portfolio_.num_instruments),
                BuyAndHold,
            )
            self.optimizers[timestamp] = optimizeStockPortfolioRiskAverse(
                portfolio_,
                logFolderPath / f"portfolioOptimization_{timestamp}.csv",
            )

    def evaluateStrategy(self):
        pass

    def aggregateResults(self):
        pass
