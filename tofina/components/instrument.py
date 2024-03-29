import torch
import tofina.utils as utils
from tofina.components.asset import Asset
from typing import Callable

payoffFnType = Callable[[torch.Tensor, int, dict], torch.Tensor]


class Instrument:
    def __init__(
        self,
        name: str,
        assetName: str,
        assetSimulation: torch.Tensor,
        payoffFn: payoffFnType,
        price: float,
        **kwargs
    ) -> None:
        self.name = name
        self.assetName = assetName
        self.assetSimulation = assetSimulation
        self.payoff = payoffFn
        self.price = torch.nn.Parameter(
            torch.tensor([price]).float(), requires_grad=False
        )
        self.params = utils.convertKwargsToTorchParameters(kwargs)
        self.revenue = self.calculateProfit()

    def updateAssetSimulation(self, newAssetSimulation: torch.Tensor) -> None:
        self.assetSimulation = newAssetSimulation
        self.revenue = self.calculateProfit()

    def calculateProfit(self) -> torch.Tensor:
        instrumentPayout = []
        processLength = self.assetSimulation.shape[-1]
        for t in range(processLength):
            # Permutes are required to make Payout function simpler
            periodPayoff = self.payoff(
                self.assetSimulation.permute(1, 0), t, self.params
            )
            instrumentPayout.append(periodPayoff)
        return torch.stack(instrumentPayout).permute(1, 0)


def OneTimeComissionDecorator(comission: float, payoffFn: payoffFnType) -> payoffFnType:
    def comissionPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
        return payoffFn(X, t, params) - comission

    return comissionPayout


def NonDerivativePayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return X[t]


def NonDerivativePayoutShort(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return X[0] + (X[0] - X[t])


def optionPayout_(
    X: torch.Tensor,
    t: int,
    params: dict,
    isCall: bool = True,
    optionType: str = "European",
) -> torch.Tensor:
    if optionType not in ["European", "American"]:
        raise ValueError("optionType must be either European or American")

    strikePrice = params["strikePrice"]
    maturity = params["maturity"] - 1

    if (optionType == "European" and t < maturity) or (t > maturity):
        return torch.zeros(X[t].shape[0])

    if isCall:
        payout = X[t] - strikePrice
    else:
        payout = strikePrice - X[t]
    payout[payout < 0] = 0

    return payout


def EuropeanCallPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=True, optionType="European")


def EuropeanPutPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=False, optionType="European")


def AmericanCallPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=True, optionType="American")


def AmericanPutPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=False, optionType="American")
