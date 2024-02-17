import yfinance as yf


def loadHistoricalStockData(ticker):
    return yf.download(ticker)
