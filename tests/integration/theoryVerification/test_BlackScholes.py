from tofina.components import preference
from tofina.theory import blackScholes
from tofina.macros import utilityEqualization
import pandas as pd


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
    utilityEqualization.DerivativePricing(
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
    utilityEqualization.DerivativePricing(
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


# TODO: Figure out implied volatility
def test_BlackScholesImpliedVolatility():
    optimizationLogFilePath = "./tests/results/BlackScholesImpliedVolatility.csv"
    S = 100
    T = 10
    r = 0.05
    K = 100 * (1 + r) ** T + 20
    sigma = 0.09
    optionCall = True
    optionPrice = blackScholes.BlackScholesOptionPricing(S, K, T, r, sigma, optionCall)
    initial_sigma = 1.0

    zeroDerivativePortfolio = blackScholes.generateBlackScholesPortfolio(
        S=S,
        K=K,
        T=T + 1,
        r=r,
        sigma=initial_sigma,
        optionCall=optionCall,
        optionPrice=optionPrice,
        initialWeights=[0.5, 0.5, 0],
        monteCarloTrials=10000,
        impliedVolaility=True,
    )
    utilityEqualization.ImpliedVolatility(
        zeroDerivativePortfolio,
        derivativeTargets=["EuropeanOption_Company"],
        optimizationLogFilePath=optimizationLogFilePath,
        utility=preference.Preference(
            moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
            timeDiscountFn=preference.NoTimeDiscount,
        ),
        derivativeWeight=0.2,
        earlyStoppingTolerance=1000000000,
        iterations=500,
        lr=0.1,
    )
