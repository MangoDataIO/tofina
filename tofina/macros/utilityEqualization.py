from tofina.components import optimizer, logger, preference, portfolio
import tofina.utils as utils
from copy import deepcopy
import torch
from typing import List, Optional, Callable


def rebalancePortfolioWeights(
    portfolio_: portfolio.Portfolio,
    initialWeights: List[float],
    derivativeTargets: List[str],
    derivativeWeight: float,
) -> torch.Tensor:
    numDerivatives = len(derivativeTargets)
    numInstruments = len(initialWeights)
    instrumentList = list(portfolio_.strategy.instruments.keys())
    newWeights = deepcopy(initialWeights)
    subtract = numDerivatives * derivativeWeight / numInstruments
    for instrument in derivativeTargets:
        i = instrumentList.index(instrument)
        newWeights[i] = derivativeWeight + subtract
    return newWeights - subtract


def derivativePricingLocationMapper(target, portfolio: portfolio.Portfolio):
    return f"portfolio.strategy.instruments['{target}'].price"


def impliedVolatilityLocationMapper(target, portfolio: portfolio.Portfolio):
    assetName = portfolio.strategy.instruments[target].assetName
    pass


def UtilityEqualization(
    zeroDerivativePortfolio: portfolio.Portfolio,
    derivativeTargets: List[str],
    optimizationLogFilePath: Optional[str],
    utility: preference.Preference,
    locationMapper: Callable,
    derivativeWeight=0.005,
    earlyStoppingTolerance: float = 0.00001,
    iterations: int = 1000,
    lr: float = 0.1,
) -> None:
    """
    New portfolio and preference derivative pricing model.
    Finds deriative price that makes person indifferent between
    adding and not adding arbitrary small ammount of the derivative
    to their portfolio
    """
    derivativePortfolioWeights = rebalancePortfolioWeights(
        zeroDerivativePortfolio,
        zeroDerivativePortfolio.strategy.normalizedWeights,
        derivativeTargets,
        derivativeWeight,
    )

    logger_ = logger.CsvLogger(filePath=optimizationLogFilePath)
    portfolioOptimizer = optimizer.Optimizer(
        portfolio=zeroDerivativePortfolio,
        preference=utility,
        logger=logger_,
    )
    targetUtility = portfolioOptimizer.utility
    portfolioOptimizer.registerLoss(
        lossTargets=["utility"],
        lossFn=optimizer.UtilityEqualizationLoss,
        targetUtility=utils.tensorToFloat(targetUtility),
    )
    zeroDerivativePortfolio.setPortfolioWeights(
        derivativePortfolioWeights, normalizeWeights=False
    )
    instrumentList = list(zeroDerivativePortfolio.strategy.instruments.keys())
    optimizationParams = []
    for target in derivativeTargets:
        index = instrumentList.index(target)
        location = f"portfolio.strategy.instruments['{target}'].price"
        optimizationParams.append(location)
        portfolioOptimizer.registerMetric(
            target,
            metricTargets=[location],
            metricFn=lambda x, params: utils.tensorToFloat(x),
        )
        portfolioOptimizer.registerMetric(
            "weight" + target,
            metricTargets=["portfolio.strategy.normalizedWeights"],
            metricFn=lambda x, params: utils.tensorToFloat(x[index]),
        )
    portfolioOptimizer.optimize(
        paramsToOptimize=optimizationParams,
        lr=lr,
        earlyStoppingTolerance=earlyStoppingTolerance,
        iterations=iterations,
    )


def DerivativePricing(
    zeroDerivativePortfolio: portfolio.Portfolio,
    derivativeTargets: List[str],
    optimizationLogFilePath: Optional[str],
    utility: preference.Preference,
    derivativeWeight=0.005,
    earlyStoppingTolerance: float = 0.00001,
    iterations: int = 1000,
    lr: float = 0.1,
) -> None:
    """
    New portfolio and preference derivative pricing model.
    Finds deriative price that makes person indifferent between
    adding and not adding arbitrary small ammount of the derivative
    to their portfolio
    """
    derivativePortfolioWeights = rebalancePortfolioWeights(
        zeroDerivativePortfolio,
        zeroDerivativePortfolio.strategy.normalizedWeights,
        derivativeTargets,
        derivativeWeight,
    )

    logger_ = logger.CsvLogger(filePath=optimizationLogFilePath)
    portfolioOptimizer = optimizer.Optimizer(
        portfolio=zeroDerivativePortfolio,
        preference=utility,
        logger=logger_,
    )
    targetUtility = portfolioOptimizer.utility
    portfolioOptimizer.registerLoss(
        lossTargets=["utility"],
        lossFn=optimizer.UtilityEqualizationLoss,
        targetUtility=utils.tensorToFloat(targetUtility),
    )
    zeroDerivativePortfolio.setPortfolioWeights(
        derivativePortfolioWeights, normalizeWeights=False
    )
    instrumentList = list(zeroDerivativePortfolio.strategy.instruments.keys())
    optimizationParams = []
    for target in derivativeTargets:
        index = instrumentList.index(target)
        location = f"portfolio.strategy.instruments['{target}'].price"
        optimizationParams.append(location)
        portfolioOptimizer.registerMetric(
            target,
            metricTargets=[location],
            metricFn=lambda x, params: utils.tensorToFloat(x),
        )
        portfolioOptimizer.registerMetric(
            "weight" + target,
            metricTargets=["portfolio.strategy.normalizedWeights"],
            metricFn=lambda x, params: utils.tensorToFloat(x[index]),
        )
    portfolioOptimizer.optimize(
        paramsToOptimize=optimizationParams,
        lr=lr,
        earlyStoppingTolerance=earlyStoppingTolerance,
        iterations=iterations,
    )


def ImpliedVolatility(
    zeroDerivativePortfolio: portfolio.Portfolio,
    derivativeTargets: List[str],
    optimizationLogFilePath: Optional[str],
    utility: preference.Preference,
    derivativeWeight=0.005,
    earlyStoppingTolerance: float = 0.00001,
    iterations: int = 1000,
    lr: float = 0.1,
) -> None:
    """
    New portfolio and preference derivative pricing model.
    Finds deriative price that makes person indifferent between
    adding and not adding arbitrary small ammount of the derivative
    to their portfolio
    """
    derivativePortfolioWeights = rebalancePortfolioWeights(
        zeroDerivativePortfolio,
        zeroDerivativePortfolio.strategy.normalizedWeights,
        derivativeTargets,
        derivativeWeight,
    )

    logger_ = logger.CsvLogger(filePath=optimizationLogFilePath)
    portfolioOptimizer = optimizer.Optimizer(
        portfolio=zeroDerivativePortfolio,
        preference=utility,
        logger=logger_,
    )
    targetUtility = portfolioOptimizer.utility
    portfolioOptimizer.registerLoss(
        lossTargets=["utility"],
        lossFn=optimizer.UtilityEqualizationLoss,
        targetUtility=utils.tensorToFloat(targetUtility),
    )
    zeroDerivativePortfolio.setPortfolioWeights(
        derivativePortfolioWeights, normalizeWeights=False
    )
    instrumentList = list(zeroDerivativePortfolio.strategy.instruments.keys())
    optimizationParams = []
    for target in derivativeTargets:
        index = instrumentList.index(target)
        location = f"portfolio.strategy.instruments['{target}'].price"
        optimizationParams.append(location)
        portfolioOptimizer.registerMetric(
            target,
            metricTargets=[location],
            metricFn=lambda x, params: utils.tensorToFloat(x),
        )
        portfolioOptimizer.registerMetric(
            "weight" + target,
            metricTargets=["portfolio.strategy.normalizedWeights"],
            metricFn=lambda x, params: utils.tensorToFloat(x[index]),
        )
    portfolioOptimizer.optimize(
        paramsToOptimize=optimizationParams,
        lr=lr,
        earlyStoppingTolerance=earlyStoppingTolerance,
        iterations=iterations,
    )
