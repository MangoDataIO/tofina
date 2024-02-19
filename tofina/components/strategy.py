import torch
import tofina.utils as utils
from typing import List, Callable, Mapping
from tofina.components import instrument

instrumentsDictType = Mapping[str, instrument.Instrument]
liquidationFnType = Callable[[torch.Tensor, instrumentsDictType, dict], torch.Tensor]


class Strategy:
    def __init__(
        self,
        portfolioWeights: List[float],
        liquidationFn: liquidationFnType,
        instruments: instrumentsDictType,
        normalizeWeights: bool = False,
        cache_liquidations: bool = False,
        cache_returns: bool = False,
        **kwargs
    ) -> None:
        self.liquidationFn = liquidationFn
        self.instruments = instruments

        self.softmax = torch.nn.Softmax(dim=0)
        self.setPortfolioWeights(portfolioWeights, normalizeWeights=normalizeWeights)
        self.params = utils.convertKwargsToTorchParameters(kwargs)
        self.cache_liquidations = cache_liquidations
        self.cache_returns = cache_returns
        self.liquidations_cache = None
        self.returns_cache = None

    @property
    def normalizedWeights(self) -> torch.Tensor:
        return self.softmax(self.portfolioWeights)

    def setPortfolioWeights(
        self, portfolioWeights: List[float], normalizeWeights: bool = False
    ) -> None:
        if type(portfolioWeights) is not torch.Tensor:
            portfolioWeights = torch.tensor(portfolioWeights)
        if not normalizeWeights:
            portfolioWeights = utils.softmaxInverse(portfolioWeights)
        self.portfolioWeights = torch.nn.Parameter(
            portfolioWeights, requires_grad=False
        )

    def backrollLiquidations(self, liquidations: torch.Tensor) -> torch.Tensor:
        """
        Estimating how much instruments were liquidated each period.
        The most problematic part is to estimate how much instument
        is remaining after previous liquidations.
        """
        backroll = (1 - liquidations).cumprod(axis=2)
        return liquidations[:, :, 1:] * backroll[:, :, :-1]

    def liquidations_(self, assetX: torch.Tensor) -> torch.Tensor:
        liquidations = self.liquidationFn(assetX, self.instruments, self.params)
        liquidations[:, :, -1] = 1
        liquidations[:, :, 0] = 0
        _, _, processLength = liquidations.shape
        for i, key in enumerate(self.instruments):
            if (
                "maturity" in self.instruments[key].params
                and self.instruments[key].params["maturity"] <= processLength
            ):
                liquidations[i, :, self.instruments[key].params["maturity"] - 1] = 1
        liquidations = self.backrollLiquidations(liquidations).permute(1, 2, 0)
        return liquidations

    def returns_(self, instrumentX: torch.Tensor) -> torch.Tensor:
        prices = torch.cat(
            [instrument.price for instrument in self.instruments.values()]
        )
        returns = (instrumentX.permute(1, 2, 0) - prices) / prices
        return returns

    def liquidations(self, assetX: torch.Tensor) -> torch.Tensor:
        if self.cache_liquidations:
            if self.liquidations_cache is None:
                self.liquidations_cache = self.liquidations_(assetX)
            return self.liquidations_cache
        else:
            return self.liquidations_(assetX)

    def returns(self, instrumentX: torch.Tensor) -> torch.Tensor:
        if self.cache_returns:
            if self.returns_cache is None:
                self.returns_cache = self.returns_(instrumentX)
            return self.returns_cache
        else:
            return self.returns_(instrumentX)

    def estimateProfit(
        self, assetX: torch.Tensor, instrumentX: torch.Tensor
    ) -> torch.Tensor:
        liquidations = self.liquidations(assetX)
        returns = self.returns(instrumentX)

        return (self.normalizedWeights * liquidations * returns[:, 1:, :]).sum(axis=2)


def BuyAndHold(
    Xt: torch.Tensor, instruments: instrumentsDictType, params: dict
) -> torch.Tensor:
    instrumentsNum = len(instruments)
    processLength = Xt.shape[-1]
    monteCarloTrials = Xt.shape[-2]

    liquidations = torch.zeros((instrumentsNum, monteCarloTrials, processLength))
    liquidations[:, :, -1] = 1
    return liquidations


def UniformLiquidation(
    Xt: torch.Tensor, instruments: instrumentsDictType, params: dict
) -> torch.Tensor:
    instrumentsNum = len(instruments)
    processLength = Xt.shape[-1]
    monteCarloTrials = Xt.shape[-2]
    liquidations = torch.zeros((instrumentsNum, monteCarloTrials, processLength))
    liquidations[:, :, :] = 1 / processLength
    liquidations[:, :, -1] = 1
    return liquidations
