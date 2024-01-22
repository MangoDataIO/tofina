from tofina.components import asset, instrument, strategy, portfolio
from tofina import utils


def test_PortfolioObj():
    portfolio_ = portfolio.Portfolio(processLength=10, monteCarloTrials=1000)
    portfolio_.addAsset(
        name="FakeCompany",
        processFn=asset.CompanyValueNormalDistributionProcess,
        mean=0.1,
        std=0.2,
        initialValue=100,
    )
    portfolio_.addAsset(
        name="FakeGovernmentObligation",
        processFn=asset.GovernmentObligtaionProcess,
        initialValue=100,
        interestRate=0.05,
    )

    portfolio_.addInstrument(
        assetName="FakeCompany",
        name="Stock",
        payoffFn=instrument.NonDerivativePayout,
        price=100,
    )

    portfolio_.addInstrument(
        assetName="FakeGovernmentObligation",
        name="Bond",
        payoffFn=instrument.NonDerivativePayout,
        price=100,
    )

    portfolio_.addInstrument(
        assetName="FakeCompany",
        name="PutOption",
        payoffFn=instrument.EuropeanPutPayout,
        price=10,
        strikePrice=100,
        maturity=10,
    )

    portfolio_.setStrategy(
        portfolioWeights=[0.6, 0.4, 0],
        liquidationFn=strategy.BuyAndHold,
    )

    test = portfolio_.simulatePnL()
    correct_value = (
        0.6 * (portfolio_.instruments["Stock_FakeCompany"].revenue - 100) / 100
        + 0.4
        * (portfolio_.instruments["Bond_FakeGovernmentObligation"].revenue - 100)
        / 100
    )[:, -1]

    assert utils.check_equality(correct_value, test[:, -1])

    portfolio_.setStrategy(
        portfolioWeights=[0.0, 1, 0.0],
        liquidationFn=strategy.BuyAndHold,
    )
    test = portfolio_.simulatePnL()
    assert utils.check_equality(1.05**9 - 1, test[:, -1])
