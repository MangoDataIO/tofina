from tofina.components import asset, instrument
import torch


def test_allInstruments():
    stockName = "testStock"
    testAsset = asset.Asset(
        stockName,
        asset.CompanyValueNormalDistributionProcess,
        10,
        1000,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )

    instrument1 = instrument.Instrument(
        name="stockInstrument",
        assetName=stockName,
        assetSimulation=testAsset.monteCarloSimulation,
        payoffFn=instrument.NonDerivativePayout,
        price=100,
    )

    instrument2 = instrument.Instrument(
        name="EuropeanCallInstrument",
        assetName=stockName,
        assetSimulation=testAsset.monteCarloSimulation,
        payoffFn=instrument.EuropeanCallPayout,
        price=0,
        strikePrice=100,
        maturity=5,
    )

    instrument3 = instrument.Instrument(
        name="EuropeanPutInstrument",
        assetName=stockName,
        assetSimulation=testAsset.monteCarloSimulation,
        payoffFn=instrument.EuropeanPutPayout,
        price=0,
        strikePrice=100,
        maturity=5,
    )

    instrument4 = instrument.Instrument(
        name="AmericanCallInstrument",
        assetName=stockName,
        assetSimulation=testAsset.monteCarloSimulation,
        payoffFn=instrument.AmericanCallPayout,
        price=0,
        strikePrice=100,
        maturity=5,
    )

    instrument5 = instrument.Instrument(
        name="AmericanPutInstrument",
        assetName=stockName,
        assetSimulation=testAsset.monteCarloSimulation,
        payoffFn=instrument.AmericanPutPayout,
        price=0,
        strikePrice=100,
        maturity=5,
    )

    assert (instrument1.revenue == testAsset.monteCarloSimulation).all()
    assert instrument2.revenue[:, 5:].sum() == 0
    assert instrument3.revenue[:, 5:].sum() == 0
    assert instrument4.revenue[:, 5:].sum() == 0
    assert instrument5.revenue[:, 5:].sum() == 0
    assert instrument2.revenue[:, :4].sum() == 0
    assert instrument3.revenue[:, :4].sum() == 0
    assert instrument4.revenue[:, :4].sum() > 0
    assert instrument5.revenue[:, :4].sum() > 0
    assert instrument2.revenue[:, 4].sum() > 0
    assert instrument3.revenue[:, 4].sum() > 0
    assert instrument2.revenue[:, 4].sum() == instrument4.revenue[:, 4].sum()
    assert instrument3.revenue[:, 4].sum() == instrument5.revenue[:, 4].sum()
    assert (
        instrument2.revenue[instrument2.revenue > 0]
        == testAsset.monteCarloSimulation[instrument2.revenue > 0] - 100
    ).all()
    assert (
        instrument3.revenue[instrument3.revenue > 0]
        == 100 - testAsset.monteCarloSimulation[instrument3.revenue > 0]
    ).all()
    assert (
        instrument4.revenue[instrument4.revenue > 0]
        == testAsset.monteCarloSimulation[instrument4.revenue > 0] - 100
    ).all()
    assert (
        instrument5.revenue[instrument5.revenue > 0]
        == 100 - testAsset.monteCarloSimulation[instrument5.revenue > 0]
    ).all()

    assert instrument2.revenue[testAsset.monteCarloSimulation < 100].sum() == 0
    assert instrument2.revenue[testAsset.monteCarloSimulation > 100].sum() > 0
    assert instrument4.revenue[testAsset.monteCarloSimulation < 100].sum() == 0
    assert instrument4.revenue[testAsset.monteCarloSimulation > 100].sum() > 0

    assert instrument3.revenue[testAsset.monteCarloSimulation < 100].sum() > 0
    assert instrument3.revenue[testAsset.monteCarloSimulation > 100].sum() == 0
    assert instrument5.revenue[testAsset.monteCarloSimulation < 100].sum() > 0
    assert instrument5.revenue[testAsset.monteCarloSimulation > 100].sum() == 0
