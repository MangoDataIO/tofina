import torch
import tofina.utils as utils
from typing import Callable

payoffFnType = Callable[[torch.Tensor, int, dict], torch.Tensor]


class Instrument:
    def __init__(
        self,
        name: str,
        assetSimulation: torch.Tensor,
        payoffFn: payoffFnType,
        price: float,
        short: bool = False,
        comission: float = 0,
        **kwargs
    ) -> None:
        self.name = name
        self.assetSimulation = assetSimulation
        self.payoff = payoffFn
        self.price = torch.nn.Parameter(
            torch.tensor([price]).float(), requires_grad=False
        )
        self.params = utils.convertKwargsToTorchParameters(kwargs)
        self.comission = torch.nn.Parameter(
            torch.tensor([comission]), requires_grad=False
        )
        self.short = short
        self.revenue = self.calculateProfit()

    def calculateProfit(self) -> torch.Tensor:
        instrumentPayout = []
        payoffSign = -1 if self.short else 1
        processLength = self.assetSimulation.shape[-1]
        for t in range(processLength):
            periodPayoff = payoffSign * self.payoff(
                self.assetSimulation.permute(1, 0), t, self.params
            )
            instrumentPayout.append(periodPayoff)
        return torch.stack(instrumentPayout).permute(1, 0)


def NonDerivativePayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return X[t]


def optionPayout_(
    X: torch.Tensor,
    t: int,
    params: dict,
    isCall: bool = True,
    optionType: str = "European",
) -> torch.Tensor:
    strikePrice = params["strikePrice"]
    maturity = params["maturity"] - 1
    if isCall:
        payout = X[t] - strikePrice
    else:
        payout = strikePrice - X[t]
    payout[payout < 0] = 0
    if optionType == "European":
        payout[t != maturity] = 0
    elif optionType == "American":
        payout[t > maturity] = 0
    else:
        raise ValueError("optionType must be either European or American")
    return payout


def EuropeanCallPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=True, optionType="European")


def EuropeanPutPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=False, optionType="European")


def AmericanCallPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=True, optionType="American")


def AmericanPutPayout(X: torch.Tensor, t: int, params: dict) -> torch.Tensor:
    return optionPayout_(X, t, params, isCall=False, optionType="American")
