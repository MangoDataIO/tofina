from tofina.components import portfolio, instrument, asset, strategy
from typing import List, Optional, Union, Mapping
import torch


def generateStockPortfolioFromMultiAsset(
    assetName: Union[str, List[str]],
    processFn: asset.processFnType,
    prices: torch.Tensor,
    processParams: dict = {},
    processLength: int = 2,
    monteCarloTrials: int = 100000,
):
    portfolio_ = portfolio.Portfolio(
        processLength=processLength, monteCarloTrials=monteCarloTrials
    )
    portfolio_.addAsset(name=assetName, processFn=processFn, **processParams)
    for i, asset in enumerate(assetName):
        portfolio_.addInstrument(
            assetName=asset,
            name="Stock",
            payoffFn=instrument.NonDerivativePayout,
            price=prices[i],
        )
    portfolio_.setStrategy(
        portfolioWeights=[1.0 / len(assetName)] * len(assetName),
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    return portfolio_
