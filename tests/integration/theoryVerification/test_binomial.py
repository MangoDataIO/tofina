from tofina.components import asset, preference
from tofina.theory import binomial
import tofina.utils as utils
from tofina.macros import portfolioOptimization, utilityEqualization
import pandas as pd
import numpy as np


def test_Theory():
    params = dict(S0=100, u=1.2, d=0.9, R=0.1)
    qU = binomial.BinomialTwoPeriodTheory(**params).qU
    assert qU > 0.66 and qU < 0.67
    X = asset.CompanyValueBinomialProcess(
        2, 100, dict(S0=100, u=1.2, d=0.9, R=0.1, initialWeights=[0.5, 0.5], qU=0.9)
    )
    assert X[:, 1].mean() > 110 and X[:, 1].mean() < 120


def test_TwoPeriod():
    martingaleFilePath = "./tests/results/binomialTwoPeriodMartingale.csv"
    bondAdvantageFilePath = "./tests/results/binomialTwoPeriodBondAdvantage.csv"
    stockAdvantageFilePath = "./tests/results/binomialTwoPeriodStockAdvantage.csv"
    portfolio_ = binomial.generateBinomialPortfolio(qU=0.75)
    optimStock = portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, stockAdvantageFilePath
    )
    portfolio_ = binomial.generateBinomialPortfolio(qU=0.5)
    optimBond = portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, bondAdvantageFilePath
    )
    portfolio_ = binomial.generateBinomialPortfolio()
    optimMartingale = portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, martingaleFilePath
    )
    assert (
        np.abs(
            (
                optimMartingale.optimizationResult["final_loss"]
                - optimMartingale.optimizationResult["initial_loss"]
            )
        )
        < 0.01
    )
    assert optimBond.optimizationResult["final_weightBond"] > 0.99
    assert optimStock.optimizationResult["final_weightStock"] > 0.99


def test_TwoPeriodMartingaleEuropeanOptions():
    martingaleFilePath = "./tests/results/binomialTwoPeriodOptionMartingale.csv"
    optionAdvantageFilePath = "./tests/results/binomialTwoPeriodOptionAdvantage.csv"
    optionDisAdvantageFilePath = (
        "./tests/results/binomialTwoPeriodOptionDisAdvantage.csv"
    )
    portfolio_ = binomial.generateBinomialPortfolio(
        optionCall=True, optionStrikePrice=100, initialWeights=[0.4, 0.4, 0.2]
    )
    portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, martingaleFilePath, optionInPortfolio=True
    )
    portfolio_ = binomial.generateBinomialPortfolio(
        optionCall=True,
        optionStrikePrice=100,
        optionPrice=11,
        initialWeights=[0.4, 0.4, 0.2],
    )
    portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, optionAdvantageFilePath, optionInPortfolio=True
    )
    portfolio_ = binomial.generateBinomialPortfolio(
        optionCall=True,
        optionStrikePrice=100,
        optionPrice=13,
        initialWeights=[0.4, 0.4, 0.2],
    )
    portfolioOptimization.optimizeStockBondPortfolioRiskNeutral(
        portfolio_, optionDisAdvantageFilePath, optionInPortfolio=True
    )
    martingale = pd.read_csv(martingaleFilePath)
    optionAdvantage = pd.read_csv(optionAdvantageFilePath)
    optionDisAdvantage = pd.read_csv(optionDisAdvantageFilePath)
    assert np.abs((martingale["loss"].iloc[-1] - martingale["loss"].iloc[0])) < 0.01
    assert optionAdvantage["weightOption"].iloc[-1] > 0.99
    assert optionDisAdvantage["weightOption"].iloc[-1] < 0.01


def test_TwoPeriodOptionPricing():
    optimizationLogFilePath = "./tests/results/binomialTwoPeriodOptionPricing.csv"
    target = binomial.BinomialTwoPeriodTheory().priceEuropeanOption(call=False) * 100
    zeroDerivativePortfolio = binomial.generateBinomialPortfolio(
        optionCall=False,
        optionStrikePrice=100,
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
    )
    optimizationLog = pd.read_csv(optimizationLogFilePath)
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] > target - 1
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] < target + 1


def test_MultiPeriodOptionPricingPut():
    optimizationLogFilePath = "./tests/results/binomialMultiPeriodOptionPricingPut.csv"

    binomialMultiTheory = binomial.BinomialMultiPeriodTheory()
    target = (
        binomialMultiTheory.priceEuropeanOption(maturity=4, call=False, strikePrice=1.7)
        * 100
    )

    zeroDerivativePortfolio = binomial.generateBinomialPortfolio(
        optionCall=False,
        optionMaturity=5,
        optionStrikePrice=170,
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
    )
    optimizationLog = pd.read_csv(optimizationLogFilePath)
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] > target - 1
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] < target + 1


def test_MultiPeriodOptionPricingCall():
    optimizationLogFilePath = "./tests/results/binomialMultiPeriodOptionPricingCall.csv"

    binomialMultiTheory = binomial.BinomialMultiPeriodTheory()
    target = (
        binomialMultiTheory.priceEuropeanOption(maturity=9, call=True, strikePrice=1.7)
        * 100
    )

    zeroDerivativePortfolio = binomial.generateBinomialPortfolio(
        optionCall=True,
        optionMaturity=10,
        optionStrikePrice=170,
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
    )
    optimizationLog = pd.read_csv(optimizationLogFilePath)
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] > target - 1
    assert optimizationLog["EuropeanOption_Company"].iloc[-1] < target + 1


def testTheoreticalModels():
    twoPeriod = binomial.BinomialTwoPeriodTheory()
    multiPeriod = binomial.BinomialMultiPeriodTheory()
    twoPeriod.priceEuropeanOption() == multiPeriod.priceEuropeanOption(maturity=1)
