from tofina.components import asset, instrument, portfolio, strategy
import torch


def test_autoExcerciseAmericanOption():
    portfolio_ = portfolio.Portfolio(10, 1000)
    portfolio_.addAsset(
        name="testStock",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addInstrument(
        assetName="testStock",
        name="AmericanCallInstrument",
        payoffFn=instrument.AmericanCallPayout,
        price=1,
        strikePrice=100,
        maturity=5,
    )
    portfolio_.setStrategy(
        portfolioWeights=[1],
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    profit = portfolio_.simulatePnL()
    assert profit.sum() > 0


def test_autoExcerciseAmericanOption2():
    portfolio_ = portfolio.Portfolio(3, 1000)
    portfolio_.addAsset(
        name="testStock",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addInstrument(
        assetName="testStock",
        name="AmericanCallInstrument",
        payoffFn=instrument.AmericanCallPayout,
        price=1,
        strikePrice=100,
        maturity=5,
    )
    portfolio_.setStrategy(
        portfolioWeights=[1],
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    profit = portfolio_.simulatePnL()
    assert profit.sum() > 0


def test_autoExcerciseEuropeanOption():
    portfolio_ = portfolio.Portfolio(10, 1000)
    portfolio_.addAsset(
        name="testStock",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addInstrument(
        assetName="testStock",
        name="EuropeanCallInstrument",
        payoffFn=instrument.EuropeanCallPayout,
        price=1,
        strikePrice=100,
        maturity=5,
    )
    portfolio_.setStrategy(
        portfolioWeights=[1],
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    profit = portfolio_.simulatePnL()
    assert profit.sum() > 0


def test_autoExcerciseEuropeanOption2():
    portfolio_ = portfolio.Portfolio(3, 1000)
    portfolio_.addAsset(
        name="testStock",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addInstrument(
        assetName="testStock",
        name="EuropeanCallInstrument",
        payoffFn=instrument.EuropeanCallPayout,
        price=1,
        strikePrice=100,
        maturity=5,
    )
    portfolio_.setStrategy(
        portfolioWeights=[1],
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    profit = portfolio_.simulatePnL()
    assert profit.sum() < 0
