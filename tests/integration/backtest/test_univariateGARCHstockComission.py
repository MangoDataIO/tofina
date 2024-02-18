from tofina.components.backtest import Backtester
from tofina.extern.arch import forecastDecoratorGARCH
from tofina.extern.yfinance import loadHistoricalStockData
import tofina.components.instrument as instrument
import datetime as dt
import numpy as np
from pathlib import Path
import shutil


# TODO
# 1. Make forecaster compatible with Assets
# 2. Create realPath dictionary
# 3. Create backtest method
def test_univariateGARCHstockComission():
    SPY = loadHistoricalStockData("SPY")
    start_date = dt.datetime(2023, 1, 1)
    end_date = dt.datetime(2023, 1, 14)
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
    backtester.registerForecaster(forecaster, ["SPY"])
    backtester.addIntstrumentToPortfolio(
        "SPY",
        "Stock",
        instrument.OneTimeComissionDecorator(100, instrument.NonDerivativePayout),
    )
    backtester.addIntstrumentToPortfolio(
        "SPY", "Stock Short", instrument.NonDerivativePayoutShort
    )
    backtester.addDeposit(interestRate=0.05 / 252)
    dir_path = Path("./tests/results/SPY_GARCH_Comission")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=False)
    backtester.optimizeStrategy(dir_path, RiskAversion=2)
