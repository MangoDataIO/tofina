import torch
from arch import arch_model


def forecastDecoratorGARCH(df, split_date, horizon=20, simulations=1000, **arch_kwargs):
    ts = 100 * df["Close"].diff() / df["Close"].shift(1)
    ts = ts.dropna()
    am = arch_model(ts, **arch_kwargs)
    res = am.fit(update_freq=5, last_obs=split_date)
    simulation = res.forecast(
        method="simulation", horizon=horizon, simulations=simulations
    )

    def forecast(date: str, asset: str):
        index = simulation.mean.index.get_loc(date)
        return torch.from_numpy(simulation.simulations.values[index])

    return forecast
