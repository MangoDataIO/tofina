from tofina.components.backtest import Backtester
import tofina.components.instrument as instrument
from tofina.extern.pytorch_forecasting import forecastDecoratorDeepAR
from tofina.extern.yfinance import loadHistoricalStockData
import datetime as dt
from pathlib import Path
import shutil
import pandas as pd
import numpy as np


def create_logdir(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dir_path = Path(log_dir / "OptimizationResults")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=False)
    train_dir = Path(log_dir / "TrainingResults")
    optimizer_dir = dir_path
    return train_dir, optimizer_dir


def get_timestamps(stock_data, start_date, end_date):
    first_stock = list(stock_data.keys())[0]
    timestamps = stock_data[first_stock].index[
        np.logical_and(
            stock_data[first_stock].index >= start_date,
            stock_data[first_stock].index < end_date,
        )
    ]
    return timestamps


def register_option(df_path, ticker, backtester, start_date, end_date):
    df = pd.read_csv(df_path)
    df["Date"] = df["Date"].apply(lambda x: x.replace(" ", ""))
    df["Expiry Date"] = df["Expiry Date"].apply(lambda x: x.replace(" ", ""))
    df["Date"] = pd.to_datetime(df["Date"])
    df["Expiry Date"] = pd.to_datetime(df["Expiry Date"])
    df = df[df["Date"] >= start_date]
    df = df[df["Date"] < end_date]
    backtester.parseOptionData(df, ticker, sampleOption=100)


def DeepAR_OptionTrading_BackTest(
    start_date,
    end_date,
    log_dir,
    horizon,
    simulations,
    stocks,
    options,
    daily_interest_rate,
):

    split_date = start_date - dt.timedelta(days=1)
    stock_data = {stock: loadHistoricalStockData("AAPL") for stock in stocks}

    print("Tofina: insihed loading stock data from Yahoo Finance")

    timestamps = get_timestamps(stock_data, start_date, end_date)
    train_dir, optimize_dir = create_logdir(log_dir)

    print("Tofina: Finished creating log directories")

    forecaster = forecastDecoratorDeepAR(
        stock_data, split_date, train_dir, horizon, simulations, max_epochs=10
    )

    print("Tofina: Finished fitting DeepVAR forecaster to stock data")

    backtester = Backtester(timestamps=timestamps)

    for stock in stock_data:
        backtester.stockDataFromDataFrame(stock_data[stock], stock)
    backtester.registerForecaster(forecaster, stocks)
    backtester.addDeposit(interestRate=daily_interest_rate)

    for stock in stock_data:
        backtester.addIntstrumentToPortfolio(
            stock, f"{stock}_Stock", instrument.NonDerivativePayout
        )
        backtester.addIntstrumentToPortfolio(
            stock, f"{stock}_Stock_Short", instrument.NonDerivativePayoutShort
        )

    for option in options:
        register_option(options[option], option, backtester, start_date, end_date)

    print("Tofina: Stared portfolio optimization and backtesting")

    backtester.optimizeStrategy(
        optimize_dir, RiskAversion=0.9, stop=False, iterations=5000
    )
    results = backtester.evaluateStrategy()
    print("Tofina: Optimization Completed!")
    return backtester, results, timestamps
