import torch
import tofina.utils as utils
from tqdm import tqdm
from tofina.components import (
    portfolio,
    preference,
    logger,
)
from typing import Optional, List, Callable

MetricFnType = Callable[[List[torch.Tensor], dict], torch.Tensor]
EmptyLogger = logger.Logger()


class Optimizer:
    def __init__(
        self,
        portfolio: portfolio.Portfolio,
        preference: preference.Preference,
        logger: Optional[logger.Logger] = None,
        **kwargs
    ) -> None:
        self.portfolio = portfolio
        self.preference = preference
        self.calculateUtility()
        (
            self.availableLossTargets,
            self.availableOptimizationTargets,
        ) = iterateOptimizerParams(self)
        self.params = utils.convertKwargsToTorchParameters(kwargs)
        self.metrics = {}

        self.logger = logger
        if logger is None:
            self.logger = EmptyLogger
        self.initialMetrics = {}
        self.finalMetrics = {}

    def calculateUtility(self) -> None:
        self.revenue = self.portfolio.simulatePnL()
        self.profits = self.revenue.sum(axis=1)
        self.utility = self.preference.utility(self.revenue)

    def initiateOptimizer(
        self, paramsToOptimize: List[str], optimizer: torch.optim.Optimizer, lr: float
    ) -> torch.optim.Optimizer:
        params = []
        for param in paramsToOptimize:
            params.append(eval("self." + param))
        for param in params:
            param.requires_grad = True
        optim = optimizer(params, lr=lr)
        return optim

    def registerLoss(
        self, lossTargets: List[str], lossFn: MetricFnType, **kwargs
    ) -> None:
        self.lossTargets = lossTargets
        self.lossFn = lossFn
        for param in kwargs:
            self.params[param] = kwargs[param]

    def registerMetric(
        self, name: str, metricTargets: List[str], metricFn: MetricFnType
    ) -> None:
        self.metrics[name] = {
            "targets": metricTargets,
            "fn": metricFn,
        }

    def applyLoss(self, lossTargets: List[str], lossFn: MetricFnType) -> torch.Tensor:
        lossTargets_ = []
        for target in lossTargets:
            lossTargets_.append(eval("self." + target))
        return lossFn(*lossTargets_, self.params)

    def logMetrics(self, loss: torch.Tensor, t: int = 0) -> None:
        if self.logger is not None:
            metricDict = {}
            for metric in self.metrics:
                metricDict[metric] = self.applyLoss(
                    self.metrics[metric]["targets"],
                    self.metrics[metric]["fn"],
                )
            metricDict["loss"] = utils.tensorToFloat(loss.detach())
            self.logger.processRecord(metricDict, t)
        return metricDict

    def calculateAndLogLoss(self, t: int = 0) -> None:
        self.calculateUtility()
        loss = self.applyLoss(self.lossTargets, self.lossFn)
        metricDict = self.logMetrics(loss, t)
        return loss, metricDict

    def optimize(
        self,
        paramsToOptimize: List[str],
        iterations: int = 1000,
        lr: float = 0.01,
        earlyStoppingTolerance: float = 0.00001,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
    ) -> bool:
        converged = False
        earlyStop = EarlyStopping(tol=earlyStoppingTolerance)
        optim = self.initiateOptimizer(paramsToOptimize, optimizer, lr)
        with torch.no_grad():
            loss, metricDict = self.calculateAndLogLoss(-1)
            resultDict = {
                "initial_" + metric: metricDict[metric] for metric in metricDict
            }

        for i in tqdm(range(iterations)):
            optim.zero_grad()
            loss, metricDict = self.calculateAndLogLoss(i)
            loss.backward()
            optim.step()
            if earlyStop(loss):
                converged = True
                break
        with torch.no_grad():
            loss, metricDict = self.calculateAndLogLoss(i + 1)
            for metric in metricDict:
                resultDict["final_" + metric] = metricDict[metric]
            resultDict["converged"] = converged
            self.optimizationResult = resultDict
        return resultDict


class EarlyStopping:
    def __init__(self, patience=20, tol=0.00001):
        self.patience = patience
        self.best_loss = 999999999999999
        self.noImprovement = 0
        self.tol = tol

    def __call__(self, loss):
        if loss + self.tol < self.best_loss:
            self.best_loss = loss
            self.noImprovement = 0
        else:
            self.noImprovement += 1
        if self.noImprovement == self.patience:
            return True
        else:
            return False


def PortfolioOptimizationLoss(averageUtility, params):
    return -averageUtility


def UtilityEqualizationLoss(averageUtility, params):
    targetUtility = params["targetUtility"]
    return torch.abs(averageUtility - targetUtility)


def iterateOptimizerParams(optimizer_: Optimizer) -> List[List[str]]:
    lossTargets = []
    optimizationTargets = []
    queue = [""]
    while queue:
        elem = queue.pop()
        if elem != "":
            obj = eval("optimizer_" + elem)
        else:
            obj = optimizer_
        # Order of if statements matters
        if isinstance(obj, dict):
            for i in obj:
                if type(i) is str:
                    queue.append(elem + "['" + i + "']")
                else:
                    # Else tuple
                    queue.append(elem + "[" + str(i) + "]")

        elif isinstance(obj, torch.nn.parameter.Parameter):
            optimizationTargets.append(elem)
            lossTargets.append(elem)
        elif isinstance(obj, torch.nn.Module):
            optimizationTargets.append(elem)
            lossTargets.append(elem)
        elif isinstance(obj, torch.Tensor):
            lossTargets.append(elem)
        else:
            try:
                for i in vars(obj):
                    queue.append(elem + "." + i)
                all_properties = {
                    k: v for k, v in vars(type(obj)).items() if isinstance(v, property)
                }
                for i in all_properties:
                    queue.append(elem + "." + i)
            except:
                continue
    return utils.removeTrailingSymbol(lossTargets), utils.removeTrailingSymbol(
        optimizationTargets
    )
