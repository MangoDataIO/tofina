import torch
import tofina.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Callable, Union, List

processFnType = Callable[[int, int, dict], torch.Tensor]


class Asset:
    """
    Denotes an underlying asset behind financial instruments.
    In an obvious case of options, company value is an underlying asset
    and option is an instrument (see ./instrument.py)
    Stock is also an insturment that maps to company value asset one-to-one.
    """

    def __init__(
        self,
        name: Union[str, List[str]],
        processFn: processFnType,
        processLength: int,
        monteCarloTrials: int,
        **kwargs
    ) -> None:
        self.processFn = processFn
        self.processLength = processLength
        self.monteCarloTrials = monteCarloTrials
        self.params = utils.convertKwargsToTorchParameters(kwargs)
        self.monteCarloSimulation = self.simulate()
        if len(self.monteCarloSimulation.shape) == 3:
            assert type(name) is list
            name = tuple(name)
        if len(self.monteCarloSimulation.shape) == 2:
            assert type(name) is str
        self.name = name

    def simulate(self) -> torch.Tensor:
        return self.processFn(self.processLength, self.monteCarloTrials, self.params)


def CompanyValueNormalDistributionProcess(
    processLength: int, monteCarloTrials: int, params: dict
) -> torch.Tensor:
    mean = params["mean"]
    std = params["std"]
    initalValue = params["initialValue"]
    # reparametrization trick to make differentiable
    sample = torch.randn((monteCarloTrials, processLength)) * std + mean
    X = 1 + sample
    X[:, 0] = 1
    return initalValue * X.cumprod(axis=1)


def CompanyValueMultiNormalDistributionProcess(
    processLength: int, monteCarloTrials: int, params: dict
) -> torch.Tensor:
    mean = params["mean"]
    std = params["covarianceMatrix"]
    initialValue = params["initialValue"]
    X = 1 + MultivariateNormal(mean, std).sample(
        (monteCarloTrials, processLength)
    ).permute(2, 0, 1)
    X[:, :, 0] = 1
    for i in range(len(initialValue)):
        X[i] = initialValue[i] * X[i].cumprod(axis=1)
    return X


def CompanyValueBinomialProcess(
    processLength: int, monteCarloTrials: int, params: dict
) -> torch.Tensor:
    S0 = params["S0"]
    u = params["u"]
    d = params["d"]
    qU = params["qU"]
    X = torch.zeros(monteCarloTrials, processLength)
    X[:, 0] = S0
    for i in range(1, processLength):
        X[:, i] = X[:, i - 1] * torch.where(torch.rand(monteCarloTrials) < qU, u, d)
    return X


def GovernmentObligtaionProcess(
    processLength: int, monteCarloTrials: int, params: dict
):
    initialValue = params["initialValue"]
    interestRate = params["interestRate"]
    X: torch.Tensor = initialValue * (1 + interestRate) ** torch.arange(processLength)
    return X.repeat(monteCarloTrials).reshape(monteCarloTrials, -1)
