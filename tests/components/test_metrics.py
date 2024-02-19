from tofina.components import (
    asset,
    instrument,
    strategy,
    preference,
    optimizer,
    portfolio,
    metrics,
)
import tofina.utils as utils
import pandas as pd


def test_metrics_integrity():
    portfolio_ = portfolio.Portfolio(processLength=10, monteCarloTrials=1000)
    portfolio_.addAsset(
        name="FakeCompany",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.1,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addAsset(
        name="FakeGovernmentObligation",
        processFn=asset.GovernmentObligtaionProcess,
        initialValue=100,
        interestRate=0.05,
    )

    portfolio_.addInstrument(
        assetName="FakeCompany",
        name="Stock",
        payoffFn=instrument.NonDerivativePayout,
        price=100,
    )

    portfolio_.addInstrument(
        assetName="FakeGovernmentObligation",
        name="Bond",
        payoffFn=instrument.NonDerivativePayout,
        price=100,
    )

    portfolio_.addInstrument(
        assetName="FakeCompany",
        name="PutOption",
        payoffFn=instrument.EuropeanPutPayout,
        price=10,
        strikePrice=100,
        maturity=10,
    )

    portfolio_.setStrategy(
        portfolioWeights=[0.6, 0.4, 0],
        liquidationFn=strategy.BuyAndHold,
    )

    utility = preference.Preference(
        moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
        timeDiscountFn=preference.NoTimeDiscount,
        interestRate=0.05,
        RiskAversion=0.5,
    )

    portfolioOptimizer = optimizer.Optimizer(
        portfolio=portfolio_,
        preference=utility,
        logger=None,
    )

    portfolioOptimizer.registerLoss(
        lossTargets=["utility"],
        lossFn=optimizer.PortfolioOptimizationLoss,
    )

    portfolioOptimizer.registerMetric(
        "weightStock",
        metricTargets=["portfolio.strategy.normalizedWeights"],
        metricFn=lambda x, params: utils.tensorToFloat(x[0]),
    )
    portfolioOptimizer.registerMetric(
        "weightBond",
        metricTargets=["portfolio.strategy.normalizedWeights"],
        metricFn=lambda x, params: utils.tensorToFloat(x[1]),
    )
    portfolioOptimizer.registerMetric(
        "weightOption",
        metricTargets=["portfolio.strategy.normalizedWeights"],
        metricFn=lambda x, params: utils.tensorToFloat(x[2]),
    )
    portfolioOptimizer.registerMetric(
        "sharpeRatio",
        metricTargets=["profits"],
        metricFn=metrics.SharpeRatio(Rf=0.05),
    )
    portfolioOptimizer.registerMetric(
        "AverageProfit",
        metricTargets=["profits"],
        metricFn=metrics.MeanProfit(),
    )
    portfolioOptimizer.registerMetric(
        "ProfitSD",
        metricTargets=["profits"],
        metricFn=metrics.ProfitStandardDeviation(),
    )
    portfolioOptimizer.registerMetric(
        "ScenarioCount",
        metricTargets=["profits"],
        metricFn=metrics.ScenarioCount(),
    )
    results = portfolioOptimizer.optimize(
        paramsToOptimize=["portfolio.strategy.portfolioWeights"],
    )
    sharpe1 = (results["final_AverageProfit"] - 0.05) / (results["final_ProfitSD"])
    sharpe2 = results["final_sharpeRatio"]
    count = results["final_ScenarioCount"]
    assert sharpe1 == sharpe2
    assert count == 1000
