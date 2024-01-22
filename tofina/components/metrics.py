import torch


def profitPercentile(p=0.05):
    def profitPercentileFn(profits: torch.Tensor, *args, **kwargs):
        return profits.kthvalue(int(profits.shape[0] * p), dim=0).values

    return profitPercentileFn


def ValueAtRisk(p=0.05):
    return profitPercentile(p=p)


def ProfitStandardDeviation():
    def ProfitStandardDeviationFn(profits: torch.Tensor, *args, **kwargs):
        return profits.std()

    return ProfitStandardDeviationFn


def MinProfit():
    def MinProfitFn(profits: torch.Tensor, *args, **kwargs):
        return profits.max()

    return MinProfitFn


def MeanProfit():
    def MeanProfitFn(profits: torch.Tensor, *args, **kwargs):
        return profits.mean()

    return MeanProfitFn


def MaxProfit():
    def MaxProfitFn(profits: torch.Tensor, *args, **kwargs):
        return profits.max()

    return MaxProfitFn


def ScenarioCount():
    def ScenarioCountFn(profits: torch.Tensor, *args, **kwargs):
        return profits.numel()

    return ScenarioCountFn


def SharpeRatio(Rf: float):
    def SharpeRatioFn(profits: torch.Tensor, *args, **kwargs):
        return (profits.mean() - Rf) / profits.std()

    return SharpeRatioFn
