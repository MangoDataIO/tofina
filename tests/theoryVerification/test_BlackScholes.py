from tofina.components import asset, preference
from tofina.theory import binomial, blackScholes
import tofina.utils as utils
from tofina.macros import portfolioOptimization, optionPricing
import pandas as pd
import numpy as np


def test_BlackScholesOptionPricingPut():
    optimizationLogFilePath = "./tests/results/BlackScholesOptionPricingPut.csv"

    S = 100
    T = 5
    r = 0.05
    K = 100 * (1 + r) ** T - 20
    sigma = 0.2
    optionCall = False

    target = blackScholes.BlackScholesOptionPricing(S, K, T, r, sigma, optionCall)

    zeroDerivativePortfolio = blackScholes.generateBlackScholesPortfolio(
        S=S,
        K=K,
        T=T + 1,
        r=r,
        sigma=sigma,
        optionCall=optionCall,
        optionPrice=50,
        initialWeights=[0.5, 0.5, 0],
        monteCarloTrials=10000,
    )
    optionPricing.UtilityEqualizationDerivativePricing(
        zeroDerivativePortfolio,
        derivativeTargets=["EuropeanOption_Company"],
        optimizationLogFilePath=optimizationLogFilePath,
        utility=preference.Preference(
            moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
            timeDiscountFn=preference.NoTimeDiscount,
        ),
        # derivativeWeight=0.0005,
        earlyStoppingTolerance=0.00000001,
        # iterations=10000,
        lr=0.1,
    )
    optimizationLog = pd.read_csv(optimizationLogFilePath)
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] > target - 1
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] < target + 1


def test_BlackScholesOptionPricingCall():
    optimizationLogFilePath = "./tests/results/BlackScholesOptionPricingCall.csv"

    S = 100
    T = 10
    r = 0.05
    K = 100 * (1 + r) ** T + 20
    sigma = 0.2
    optionCall = True

    target = blackScholes.BlackScholesOptionPricing(S, K, T, r, sigma, optionCall)

    zeroDerivativePortfolio = blackScholes.generateBlackScholesPortfolio(
        S=S,
        K=K,
        T=T + 1,
        r=r,
        sigma=sigma,
        optionCall=optionCall,
        optionPrice=50,
        initialWeights=[0.5, 0.5, 0],
        monteCarloTrials=10000,
    )
    optionPricing.UtilityEqualizationDerivativePricing(
        zeroDerivativePortfolio,
        derivativeTargets=["EuropeanOption_Company"],
        optimizationLogFilePath=optimizationLogFilePath,
        utility=preference.Preference(
            moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
            timeDiscountFn=preference.NoTimeDiscount,
        ),
        # derivativeWeight=0.2,
        # earlyStoppingTolerance=0.00000001,
        # iterations=10000,
        lr=0.1,
    )
    optimizationLog = pd.read_csv(optimizationLogFilePath)
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] > target - 2
    # Need to improve precision of test
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] < target + 1
