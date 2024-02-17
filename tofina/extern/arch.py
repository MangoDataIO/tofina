import torch
from arch import arch_model


def prepend_tensor_with_ones(tensor: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((tensor.shape[0], tensor.shape[1], 1))
    return torch.cat((ones, tensor), 2)


def forecastDecoratorGARCH(df, split_date, horizon=20, simulations=1000, **arch_kwargs):
    df = df.copy()

    df["diff"] = 100 * df["Close"].diff() / df["Close"].shift(1)
    df = df.dropna()
    ts = df["diff"]
    am = arch_model(ts, **arch_kwargs)
    res = am.fit(update_freq=5, last_obs=split_date)
    simulation = res.forecast(
        method="simulation", horizon=horizon - 1, simulations=simulations
    )
    simulation_values = 1 + simulation.simulations.values / 100
    simulation_values = simulation_values.cumprod(axis=2)
    simulation_values = torch.from_numpy(simulation_values)
    simulation_values = prepend_tensor_with_ones(simulation_values)
    init_values = torch.from_numpy(df["Close"].values)
    simulation_values = (simulation_values.permute(1, 2, 0) * init_values).permute(
        2, 0, 1
    )

    def forecast(date: str, asset=None):
        index = simulation.mean.index.get_loc(date)
        return simulation_values[index, :, :]

    return forecast
