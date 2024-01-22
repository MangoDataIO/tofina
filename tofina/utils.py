import torch
import math
from typing import List

TOLERANCE = 0.001


def convertKwargsToTorchParameters(kwargs: dict) -> dict:
    params = {}
    for key in kwargs:
        val = kwargs[key]
        try:
            if type(val) is torch.Tensor:
                params[key] = torch.nn.Parameter(val, requires_grad=False)
            else:
                params[key] = torch.nn.Parameter(torch.tensor(val), requires_grad=False)
        except:
            params[key] = torch.tensor(val)
    return params


def check_equality(x: torch.Tensor, y: torch.Tensor) -> bool:
    return ((x - y).abs() < TOLERANCE).all()


def removeTrailingSymbol(stringList: List[str]) -> List[str]:
    return [string[1:] for string in stringList]


def tensorToFloat(tensor: torch.Tensor) -> float:
    return float(tensor.detach())


def softmaxInverse(X: torch.Tensor) -> torch.Tensor:
    return torch.log(X)


def combinations(n: int, k: int) -> float:
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
