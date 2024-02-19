from tofina.components import (
    asset,
    instrument,
    strategy,
    portfolio,
)
import tofina.utils as utils
from typing import List, Optional


class BinomialTwoPeriodTheory:
    """
    Tomas Bjork: Arbitrage Theory in Continuous Time, 4th Edition, page 8
    """

    def __init__(
        self,
        S0: float = 1,
        u: float = 1.2,
        d: float = 0.9,
        R: float = 0.1,
    ) -> None:
        self.S0 = S0
        self.u = u
        self.d = d
        self.R = R
        self.qU = self.calculateMartingaleProb()
        self.qD = 1 - self.qU

    def calculateMartingaleProb(self) -> float:
        res = (1 + self.R - self.d) / (self.u - self.d)
        print("Mattigale prob: ", res)
        return res

    def setProb(self, qU: float) -> None:
        self.qU = qU
        self.qD = 1 - qU

    def priceEuropeanOption(self, call: bool = True, strikePrice: float = 1.0) -> float:
        if call:
            payoutU = self.u - strikePrice
            payoutD = self.d - strikePrice
        else:
            payoutU = strikePrice - self.u
            payoutD = strikePrice - self.d
        if payoutU < 0:
            payoutU = 0
        if payoutD < 0:
            payoutD = 0

        return (self.qU * payoutU + self.qD * payoutD) / (1 + self.R)


class BinomialMultiPeriodTheory(BinomialTwoPeriodTheory):
    """
    Tomas Bjork: Arbitrage Theory in Continuous Time, 4th Edition, page 25
    """

    def priceEuropeanOption(
        self, maturity: int, call: bool = True, strikePrice: int = 1
    ) -> float:
        discount = (1 + self.R) ** maturity
        if call:
            contractFunction = lambda x: x - strikePrice if x - strikePrice > 0 else 0
        else:
            contractFunction = lambda x: strikePrice - x if strikePrice - x > 0 else 0

        result = 0
        for i in range(maturity + 1):
            result += (
                utils.combinations(maturity, i)
                * (self.qU**i)
                * (self.qD ** (maturity - i))
                * contractFunction(self.S0 * (self.u**i) * (self.d ** (maturity - i)))
            )
        return result / discount


def generateBinomialPortfolio(
    S0: float = 100,
    u: float = 1.2,
    d: float = 0.9,
    R: float = 0.1,
    initialWeights: List[float] = [0.5, 0.5],
    qU: Optional[float] = None,
    optionStrikePrice: Optional[float] = None,
    optionCall: Optional[bool] = None,
    optionPrice: Optional[float] = None,
    monteCarloTrials: int = 10000,
    optionMaturity: int = 2,
) -> portfolio.Portfolio:
    BinomialPortfolio = portfolio.Portfolio(
        processLength=optionMaturity, monteCarloTrials=monteCarloTrials
    )
    if qU is None:
        theoreticalCalulator = BinomialTwoPeriodTheory(S0, u, d, R)
        qU = theoreticalCalulator.qU

    BinomialPortfolio.addAsset(
        name="Company",
        processFn=asset.CompanyValueBinomialProcess,
        S0=S0,
        u=u,
        d=d,
        qU=qU,
    )
    BinomialPortfolio.addAsset(
        name="GovernmentObligation",
        processFn=asset.GovernmentObligtaionProcess,
        initialValue=S0,
        interestRate=R,
    )
    BinomialPortfolio.addInstrument(
        assetName="Company",
        name="Stock",
        payoffFn=instrument.NonDerivativePayout,
        price=S0,
    )

    BinomialPortfolio.addInstrument(
        assetName="GovernmentObligation",
        name="Bond",
        payoffFn=instrument.NonDerivativePayout,
        price=S0,
    )
    if optionStrikePrice is not None:
        if optionPrice is None:
            optionPrice = 100 * theoreticalCalulator.priceEuropeanOption(
                call=optionCall, strikePrice=optionStrikePrice / 100
            )
            print("Martingale Option Price", optionPrice)
        BinomialPortfolio.addInstrument(
            assetName="Company",
            name="EuropeanOption",
            payoffFn=(
                instrument.EuropeanCallPayout
                if optionCall
                else instrument.EuropeanPutPayout
            ),
            price=optionPrice,
            maturity=optionMaturity,
            strikePrice=optionStrikePrice,
        )

    BinomialPortfolio.setStrategy(
        portfolioWeights=initialWeights,
        liquidationFn=strategy.BuyAndHold,
        normalizeWeights=False,
    )
    return BinomialPortfolio
