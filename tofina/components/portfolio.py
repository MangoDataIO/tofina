import torch
from tofina.components import asset, instrument, strategy, cache
from typing import List, Optional, Union, Mapping
from functools import cached_property
from tofina.constants import ASSET_CACHE_KEY, INSTRUMENT_CACHE_KEY


class Portfolio:
    def __init__(
        self,
        processLength: int,
        monteCarloTrials: int,
        cache_asset: bool = False,
        cache_instrument: bool = False,
    ) -> None:
        self.assets: Mapping[Union[str, List[str]], asset.Asset] = {}
        self.instruments: Mapping[str, instrument.Instrument] = {}
        self.strategy: Optional[strategy.Strategy] = None
        self.processLength = processLength
        self.monteCarloTrials = monteCarloTrials
        self.setup_cache(cache_asset, cache_instrument)

    def setup_cache(self, cache_asset: bool, cache_instrument: bool):
        self.caclculationsCache = cache.CalculationCache()
        if cache_asset:
            self.caclculationsCache.register_key(ASSET_CACHE_KEY)
        if cache_instrument:
            self.caclculationsCache.register_key(INSTRUMENT_CACHE_KEY)

    def addAsset(
        self, name: Union[str, List[str]], processFn: asset.processFnType, **kwargs
    ) -> None:
        asset_ = asset.Asset(
            name, processFn, self.processLength, self.monteCarloTrials, **kwargs
        )
        if type(name) is list:
            name = tuple(name)

        self.assets[name] = asset_

    def getMonteCarloSimulation(self, assetName: str) -> torch.Tensor:
        if assetName in self.assets:
            return self.assets[assetName].monteCarloSimulation
        else:
            for assetList in self.assets:
                if type(assetList) is tuple and assetName in assetList:
                    position = assetList.index(assetName)
                    return self.assets[assetList].monteCarloSimulation[position, :, :]
        raise ValueError("Asset not found")

    def addInstrument(
        self,
        assetName: str,
        name: str,
        payoffFn: instrument.payoffFnType,
        price: float,
        **kwargs
    ) -> None:
        assetSimulation = self.getMonteCarloSimulation(assetName)
        instrument_ = instrument.Instrument(
            name,
            assetSimulation,
            payoffFn,
            price,
            **kwargs,
        )
        self.instruments[name + "_" + assetName] = instrument_

    @property
    def num_instruments(self):
        return len(self.instruments)

    def setStrategy(
        self,
        portfolioWeights: List[float],
        liquidationFn: strategy.liquidationFnType,
        **kwargs
    ) -> None:
        self.strategy = strategy.Strategy(
            portfolioWeights,
            liquidationFn,
            self.instruments,
            **kwargs,
        )

    def setPortfolioWeights(self, *args, **kwargs) -> None:
        self.strategy.setPortfolioWeights(*args, **kwargs)

    @property
    def assetX(self) -> torch.Tensor:
        def assetX_(self):
            return torch.stack(
                [self.assets[asset_].monteCarloSimulation for asset_ in self.assets]
            )

        return self.caclculationsCache(assetX_, ASSET_CACHE_KEY)(self)

    @property
    def instrumentX(self) -> torch.Tensor:
        def instrumentX_(self):
            return torch.stack(
                [instrument.revenue for instrument in self.instruments.values()]
            )

        return self.caclculationsCache(instrumentX_, INSTRUMENT_CACHE_KEY)(self)

    def simulatePnL(self) -> torch.Tensor:
        return self.strategy.estimateProfit(self.assetX, self.instrumentX)
