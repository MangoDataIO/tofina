import torch
from tofina.components import asset, instrument, strategy
from typing import List, Optional, Union, Mapping
from functools import cached_property


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
        self.cache_asset = cache_asset
        self.cache_instrument = cache_instrument
        self.asset_cache = None
        self.instrument_cache = None

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

    def assetX_(self):
        return torch.stack(
            [self.assets[asset_].monteCarloSimulation for asset_ in self.assets]
        )

    def instrumentX_(self):
        return torch.stack(
            [instrument.revenue for instrument in self.instruments.values()]
        )

    @property
    def assetX(self) -> torch.Tensor:
        if self.cache_asset:
            if self.asset_cache is None:
                self.asset_cache = self.assetX_()
            return self.asset_cache
        else:
            return self.assetX_()

    @property
    def instrumentX(self) -> torch.Tensor:
        if self.cache_instrument:
            if self.instrument_cache is None:
                self.instrument_cache = self.instrumentX_()
            return self.instrument_cache
        else:
            return self.instrumentX_()

    def simulatePnL(self) -> torch.Tensor:
        return self.strategy.estimateProfit(self.assetX, self.instrumentX)
