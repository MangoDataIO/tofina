from tofina.components import asset, instrument, strategy, preference, portfolio
from tofina import utils


def test_SimplePreference():
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

    profit = portfolio_.simulatePnL()
    utility = preference.Preference(
        moneyUtilityFn=preference.MoneyUtilityRiskNeutral,
        timeDiscountFn=preference.NoTimeDiscount,
        interestRate=0.05,
        RiskAversion=0.5,
    )
    assert utils.check_equality(utility.utility(profit), profit[:, -1].mean())
