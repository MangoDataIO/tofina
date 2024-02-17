import torch
from tofina.components import (
    asset,
    portfolio,
)
from tofina.macros import portfolioGenerator


def CalculateMinimumVariancePortfolio(covariance: torch.Tensor) -> torch.Tensor:
    """
    Calculate the minimum variance portfolio given covariance matrix
    https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf page 4
    """
    covarianceInverse = torch.linalg.inv(covariance)
    return covarianceInverse.sum(dim=1) / covarianceInverse.sum()


def GenerateRandomCovarianceMatrix(size: int) -> torch.Tensor:
    """
    Generate random covariance matrix
    """
    covariance = torch.randn(size, size)
    covariance = covariance @ covariance.T
    return covariance / covariance.sum()


def GenerateVarianceMinimizationPortfolio(
    mean: float,
    covarianceMatrix: torch.Tensor,
    initialValue: torch.Tensor,
    prices: torch.Tensor,
) -> portfolio.Portfolio:
    numInstruments = covarianceMatrix.shape[-1]
    companyNames = ["Company" + str(i) for i in range(numInstruments)]
    return portfolioGenerator.generateStockPortfolioFromMultiAsset(
        assetName=companyNames,
        processFn=asset.CompanyValueMultiNormalDistributionProcess,
        prices=prices,
        processParams={
            "mean": torch.ones(numInstruments) * mean,
            "covarianceMatrix": covarianceMatrix,
            "initialValue": initialValue,
        },
    )
