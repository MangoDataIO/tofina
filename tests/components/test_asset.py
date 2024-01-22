from tofina.components import asset
import torch
import numpy as np
from tofina import utils


def stock_assertion(simulated):
    assert simulated[:, -1].min() < simulated[:, 1].min()
    assert simulated[:, -1].max() > simulated[:, 1].max()
    assert simulated[:, -1].std() > simulated[:, 1].std()
    assert simulated[:, -1].mean() - simulated[:, 1].mean() > 50
    assert utils.check_equality(simulated[:, 0], 100).all()


def test_CompanyValueNormalDistributionProcess():
    processLength = 10
    monteCarloTrials = 1000
    params = {"mean": 0.065, "std": 0.2, "initialValue": 100}
    simulated = asset.CompanyValueNormalDistributionProcess(
        processLength=processLength, monteCarloTrials=monteCarloTrials, params=params
    )
    stock_assertion(simulated)


def test_CompanyValueMultiNormalDistributionProcess():
    processLength = 10
    monteCarloTrials = 1000
    params = {
        "mean": torch.tensor([0.065, 0.065]),
        "covarianceMatrix": torch.tensor([[0.1, 0.05], [0.05, 0.15]]),
        "initialValue": torch.tensor([100, 100]),
    }
    simulated = asset.CompanyValueMultiNormalDistributionProcess(
        processLength=processLength, monteCarloTrials=monteCarloTrials, params=params
    )
    assert np.corrcoef(simulated[0].flatten(), simulated[1].flatten())[0][1] > 0.25
    for i in [0, 1]:
        stock_assertion(simulated[i])


def test_GovernmentObligationProcess():
    processLength = 10
    monteCarloTrials = 1000
    params = {"interestRate": 0.05, "initialValue": 100}
    simulated = asset.GovernmentObligtaionProcess(
        processLength=processLength, monteCarloTrials=monteCarloTrials, params=params
    )
    assert utils.check_equality(simulated[:, 0], 100).all()
    assert utils.check_equality(simulated[:, -1], 100 * 1.05**9).all()


def test_AssetObj():
    testAsset = asset.Asset(
        "testStock",
        asset.CompanyValueNormalDistributionProcess,
        10,
        1000,
        mean=0.065,
        std=0.2,
        initialValue=100,
    )
    simulated = testAsset.monteCarloSimulation
    stock_assertion(simulated)

    testMultiAsset = asset.Asset(
        ["testStock", "testStock2"],
        asset.CompanyValueMultiNormalDistributionProcess,
        10,
        1000,
        mean=[0.065, 0.065],
        covarianceMatrix=[[0.1, 0.05], [0.05, 0.15]],
        initialValue=[100, 100],
    )
    simulated = testMultiAsset.monteCarloSimulation
    assert np.corrcoef(simulated[0].flatten(), simulated[1].flatten())[0][1] > 0.25
    for i in [0, 1]:
        stock_assertion(simulated[i])
