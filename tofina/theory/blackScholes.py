import numpy as np
from scipy.stats import norm
from tofina.components import portfolio, asset, instrument, strategy
from typing import List


def BlackScholesOptionPricing(
    S: float, K: float, T: int, r: float, sigma: float, optionCall: bool
) -> float:
    # https://corporatefinanceinstitute.com/resources/derivatives/black-scholes-merton-model/

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if optionCall:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def BlackScholesDeltaHedgingCallWeights(
    S: float, K: float, T: int, r: float, sigma: float
):
    # https://pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/optionbasics.pdf - page 15
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    stockWeight = norm.cdf(d1)
    bondWeight = -K * np.exp(-r * T) * norm.cdf(d2)
    return stockWeight, bondWeight


def generateBlackScholesPortfolio(
    S: float,
    K: float,
    T: int,
    r: float,
    sigma: float,
    optionCall: bool,
    optionPrice: float,
    monteCarloTrials: int,
    initialWeights: List[float],
    impliedVolaility: bool = False,
) -> portfolio.Portfolio:
    # http://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf mu=r
    BlackScholesPortfolio = portfolio.Portfolio(
        processLength=T,
        monteCarloTrials=monteCarloTrials,
        cache_asset=not impliedVolaility,
        cache_instrument=not impliedVolaility,
    )

    BlackScholesPortfolio.addAsset(
        name="Company",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=r,
        std=sigma,
        initialValue=S,
    )
    BlackScholesPortfolio.addAsset(
        name="GovernmentObligation",
        processFn=asset.GovernmentObligtaionProcess,
        initialValue=S,
        interestRate=r,
    )
    BlackScholesPortfolio.addInstrument(
        assetName="Company",
        name="Stock",
        payoffFn=instrument.NonDerivativePayout,
        price=S,
    )

    BlackScholesPortfolio.addInstrument(
        assetName="GovernmentObligation",
        name="Bond",
        payoffFn=instrument.NonDerivativePayout,
        price=S,
    )
    if optionPrice is None:
        optionPrice = BlackScholesOptionPricing(S, K, T, r, sigma, optionCall)
        print("Martingale Option Price", optionPrice)

    if optionCall:
        payoutFn = instrument.EuropeanCallPayout
    else:
        payoutFn = instrument.EuropeanPutPayout

    BlackScholesPortfolio.addInstrument(
        assetName="Company",
        name="EuropeanOption",
        payoffFn=payoutFn,
        price=optionPrice,
        maturity=T,
        strikePrice=K,
    )

    BlackScholesPortfolio.setStrategy(
        portfolioWeights=initialWeights,
        liquidationFn=strategy.BuyAndHold,
    )
    return BlackScholesPortfolio
