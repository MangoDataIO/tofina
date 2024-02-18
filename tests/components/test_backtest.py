import tofina.components.backtest as backtest


def test_numberOfTradingDaysBetweenTwoDays():
    assert backtest.numberOfTradingDaysBetweenTwoDays("2024-02-01", "2024-02-10") == 7
