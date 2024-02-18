import torch
import pandas as pd
from typing import List, Union, Dict, Callable
import datetime as dt
import tofina.components.portfolio as portfolio
from pathlib import Path
from tofina.components import asset
from tofina.components.instrument import (
    NonDerivativePayout,
    payoffFnType,
)
from tofina.components.strategy import BuyAndHold
from tofina.macros.portfolioOptimization import optimizeStockPortfolioRiskAverse
from tofina.components.optimizer import Optimizer
from tqdm import tqdm


forcastType = Callable[[str, str, int, int, dict], torch.Tensor]
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

    def addDeposit(self, interestRate: float):
        for timestamp in self.timestamps:
            portfolio_ = self.pointInTimePortfolio[timestamp]
            portfolio_.addAsset(
                name="GIC",
                processFn=asset.GovernmentObligtaionProcess,
                initialValue=100,
                interestRate=interestRate,
            )
            portfolio_.addInstrument(
                assetName="GIC",
                name="Deposit",
                payoffFn=NonDerivativePayout,
                price=100,
            )

    def addIntstrumentToPortfolio(
        self,
        ticker: str,
        assetName: str,
        payoffFn: payoffFnType = NonDerivativePayout,
    ):
        for timestamp in self.timestamps:
            portfolio_ = self.pointInTimePortfolio[timestamp]
            price = self.historicalTrajectory[ticker][timestamp][0]
            portfolio_.addInstrument(ticker, ticker + "_" + assetName, payoffFn, price)

    def registerForecaster(
        self,
        forecast: forcastType,
        supported_tickers: List[str],
    ):
        for timestamp in self.timestamps:
            portfolio_ = self.pointInTimePortfolio[timestamp]
            for ticker in supported_tickers:

                def processFn(processLength, monteCarloTrials, params):
                    return forecast(
                        timestamp, ticker, processLength, monteCarloTrials, **params
                    )

                portfolio_.addAsset(ticker, processFn)

    def optimizeStrategy(self, logFolderPath: str, RiskAversion: float = 0.5):
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
                RiskAversion,
            )

    def evaluateStrategy(self):
        pass

    def aggregateResults(self):
        pass
