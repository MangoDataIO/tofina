from tofina.components import asset, preference
from tofina.theory import varianceMinimization
import tofina.utils as utils
from tofina.macros import portfolioOptimization, optionPricing
import pandas as pd
import numpy as np
import torch


def test_VarianceMinimization():
    minVarianceFilePath1 = "./tests/results/minVariancePortfolio1.csv"
    minVarianceFilePath2 = "./tests/results/minVariancePortfolio2.csv"
    minVarianceFilePath3 = "./tests/results/minVariancePortfolio3.csv"
    covariance1 = torch.tensor([[5, 1], [1, 10]]).float() / 100
    covariance2 = torch.tensor([[5, 0, 0], [0, 10, 0], [0, 0, 15]]).float() / 100
    covariance3 = (
        torch.tensor(
            [
                [90, 70, 60, 50, 30],
                [70, 80, 30, 40, 20],
                [60, 30, 70, 20, 10],
                [50, 40, 20, 60, 10],
                [30, 20, 10, 10, 50],
            ]
        ).float()
        / 100
    )
    for i in range(5):
        for z in range(5):
            if i != z:
                covariance3[i][z] = covariance3[i][z] / 3

    mean = 0.03

    portfolio_ = varianceMinimization.GenerateVarianceMinimizationPortfolio(
        mean, covariance1, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])
    )
    portfolioOptimization.optimizeStockPortfolioRiskAverse(
        portfolio_, minVarianceFilePath1
    )
    portfolio_ = varianceMinimization.GenerateVarianceMinimizationPortfolio(
        mean, covariance2, torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])
    )
    portfolioOptimization.optimizeStockPortfolioRiskAverse(
        portfolio_, minVarianceFilePath2
    )
    portfolio_ = varianceMinimization.GenerateVarianceMinimizationPortfolio(
        mean,
        covariance3,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    portfolioOptimization.optimizeStockPortfolioRiskAverse(
        portfolio_, minVarianceFilePath3
    )

    theoreticalWeights1 = varianceMinimization.CalculateMinimumVariancePortfolio(
        covariance1
    )
    theoreticalWeights2 = varianceMinimization.CalculateMinimumVariancePortfolio(
        covariance2
    )
    theoreticalWeights3 = varianceMinimization.CalculateMinimumVariancePortfolio(
        covariance3
    )

    df1 = pd.read_csv(minVarianceFilePath1)
    df2 = pd.read_csv(minVarianceFilePath2)
    df3 = pd.read_csv(minVarianceFilePath3)

    weight1 = torch.tensor(df1.iloc[-1][:2])
    weight2 = torch.tensor(df2.iloc[-1][:3])
    weight3 = torch.tensor(df3.iloc[-1][:5])

    assert (theoreticalWeights1 - weight1).abs().mean() < 0.03
    assert (theoreticalWeights2 - weight2).abs().mean() < 0.03
    assert (theoreticalWeights3 - weight3).abs().mean() < 0.03
