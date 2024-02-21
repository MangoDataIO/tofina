import yfinance as yf


def loadHistoricalStockData(ticker, start=None, end=None):
    if start is not None and end is not None:
        return yf.download(ticker, start=start, end=end)
    else:
        return yf.download(ticker)
