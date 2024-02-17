from tofina.components.backtest import Backtester
from tofina.extern.arch import forecastDecoratorGARCH
from tofina.extern.yfinance import loadHistoricalStockData
import datetime as dt
import numpy as np
from pathlib import Path


# TODO
# 1. Make forecaster compatible with Assets
# 2. Create realPath dictionary
# 3. Create backtest method
def test_univariateGARCHstock():
    SPY = loadHistoricalStockData("SPY")
    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2024, 1, 1)
    split_date = start_date - dt.timedelta(days=1)
    timestamps = SPY.index[
        np.logical_and(SPY.index >= start_date, SPY.index < end_date)
    ]
    horizon = 20
    simulations = 1000
    arch_kwargs = dict(vol="Garch", p=3, o=0, q=3, dist="t", mean="AR", lags=5)
    forecaster = forecastDecoratorGARCH(
        SPY, split_date, horizon, simulations, **arch_kwargs
    )
    backtester = Backtester(timestamps=timestamps)
    backtester.stockDataFromDataFrame(SPY, "SPY")
    backtester.registerForecaster(forecaster, ["SPY"], allowShorting=False)
    dir_path = Path("./tests/results/SPY_GARCH")
    if dir_path.exists():
        dir_path.rmdir()
    dir_path.mkdir(parents=True, exist_ok=False)
    backtester.optimizeStrategy(dir_path)
