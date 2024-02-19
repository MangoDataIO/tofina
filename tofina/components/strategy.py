import torch
import tofina.utils as utils
from typing import List, Callable, Mapping
from tofina.components import instrument, cache
from tofina.constants import (
    LIQUIDATIONS_CACHE_KEY,
    RETURNS_CACHE_KEY,
    EFFECTIVE_RETURNS_CACHE_KEY,
)

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
        self.setup_cache(cache_liquidations, cache_returns)

    def setup_cache(self, cache_liquidations: bool, cache_returns: bool):
        self.calculationsCache = cache.CalculationCache()
        if cache_liquidations:
            self.calculationsCache.register_key(LIQUIDATIONS_CACHE_KEY)
        if cache_returns:
            self.calculationsCache.register_key(RETURNS_CACHE_KEY)
        if cache_liquidations and cache_returns:
            self.calculationsCache.register_key(EFFECTIVE_RETURNS_CACHE_KEY)

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

    def liquidations(self, assetX: torch.Tensor) -> torch.Tensor:
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

        return self.calculationsCache(liquidations_, LIQUIDATIONS_CACHE_KEY)(
            self, assetX
        )

    def returns(self, instrumentX: torch.Tensor) -> torch.Tensor:
        def returns_(self, instrumentX: torch.Tensor) -> torch.Tensor:
            prices = torch.cat(
                [instrument.price for instrument in self.instruments.values()]
            )
            returns = (instrumentX.permute(1, 2, 0) - prices) / prices
            return returns

        return self.calculationsCache(returns_, RETURNS_CACHE_KEY)(self, instrumentX)

    def effectiveReturns(
        self, assetX: torch.Tensor, instrumentX: torch.Tensor
    ) -> torch.Tensor:
        def effectiveReturns_(
            self, assetX: torch.Tensor, instrumentX: torch.Tensor
        ) -> torch.Tensor:
            liquidations = self.liquidations(assetX)
            returns = self.returns(instrumentX)
            return liquidations * returns[:, 1:, :]

        return self.calculationsCache(effectiveReturns_, EFFECTIVE_RETURNS_CACHE_KEY)(
            self, assetX, instrumentX
        )

    def estimateProfit(
        self, assetX: torch.Tensor, instrumentX: torch.Tensor
    ) -> torch.Tensor:
        effectiveReturns = self.effectiveReturns(assetX, instrumentX)
        return (self.normalizedWeights * effectiveReturns).sum(axis=2)


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
