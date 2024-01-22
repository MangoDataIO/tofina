import tofina.utils as utils
import torch
import math
from typing import Callable

moneyUtilityFnType = Callable[[torch.tensor, dict], torch.Tensor]
timeDiscountFnType = Callable[[int, dict], float]


class Preference:
    def __init__(
        self,
        moneyUtilityFn: moneyUtilityFnType,
        timeDiscountFn: timeDiscountFnType,
        **kwargs
    ) -> None:
        self.moneyUtilityFn = moneyUtilityFn
        self.timeDiscountFn = timeDiscountFn
        self.params = utils.convertKwargsToTorchParameters(kwargs)

    def utility(self, moneyX: torch.Tensor) -> torch.Tensor:
        processLength = moneyX.shape[-1]
        timeDiscounts = [
            self.timeDiscountFn(i, self.params) for i in range(1, processLength + 1)
        ]
        timeDiscounts = torch.tensor(timeDiscounts)
        aggregatedMoney = (moneyX * timeDiscounts).sum(axis=1)
        return self.moneyUtilityFn(aggregatedMoney, self.params).mean()


def InterestRateTimeDiscount(time: int, params: dict) -> float:
    return 1 / ((1 + params["interestRate"]) ** time)


def NoTimeDiscount(time: int, params: dict) -> float:
    return 1


def MoneyUtilityCRRA(money: torch.Tensor, params: dict) -> torch.Tensor:
    riskAversion = params["RiskAversion"]
    money = 1 + money
    money = torch.clip(money, min=0.01)  # to avoid negative utility
    return (money ** (1 - riskAversion) - 1) / (1 - riskAversion)


def MoneyUtilityRiskNeutral(money: torch.Tensor, params: dict) -> torch.Tensor:
    return money
