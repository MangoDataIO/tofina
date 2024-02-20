from tofina.components import optimizer, logger, preference, portfolio
import tofina.utils as utils
from typing import Optional
from copy import deepcopy
from functools import partial


def optimizeStockBondPortfolioRiskNeutral(
    portfolio_: portfolio.Portfolio,
    csvLogFilePath: Optional[str],
    optionInPortfolio: bool = False,
) -> optimizer.Optimizer:
    logger_ = logger.CsvLogger(filePath=csvLogFilePath)
    utility = preference.Preference(
        moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
        timeDiscountFn=preference.NoTimeDiscount,
    )
    portfolioOptimizer = optimizer.Optimizer(
        portfolio=portfolio_,
        preference=utility,
        logger=logger_,
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
    if optionInPortfolio:
        portfolioOptimizer.registerMetric(
            "weightOption",
            metricTargets=["portfolio.strategy.normalizedWeights"],
            metricFn=lambda x, params: utils.tensorToFloat(x[2]),
        )

    portfolioOptimizer.optimize(
        paramsToOptimize=["portfolio.strategy.portfolioWeights"],
    )
    return portfolioOptimizer


def optimizeStockPortfolioRiskAverse(
    portfolio_: portfolio.Portfolio,
    csvLogFilePath: Optional[str],
    RiskAversion: float = 0.5,
    **kwargs
) -> optimizer.Optimizer:
    logger_ = logger.CsvLogger(filePath=csvLogFilePath)
    utility = preference.Preference(
        moneyUtilityFn=preference.MoneyUtilityCRRA,
        timeDiscountFn=preference.NoTimeDiscount,
        RiskAversion=RiskAversion,
    )
    portfolioOptimizer = optimizer.Optimizer(
        portfolio=portfolio_,
        preference=utility,
        logger=logger_,
    )
    portfolioOptimizer.registerLoss(
        lossTargets=["utility"],
        lossFn=optimizer.PortfolioOptimizationLoss,
    )
    for i, instrument in enumerate(portfolio_.instruments):

        def metricFn(x, params, index):
            return utils.tensorToFloat(x[index])

        portfolioOptimizer.registerMetric(
            instrument + "weight",
            metricTargets=["portfolio.strategy.normalizedWeights"],
            metricFn=partial(metricFn, index=i),
        )
    portfolioOptimizer.optimize(
        paramsToOptimize=["portfolio.strategy.portfolioWeights"],
        earlyStoppingTolerance=1e-8,
        **kwargs
    )
    return portfolioOptimizer
