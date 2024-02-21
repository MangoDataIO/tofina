from pytorch_lightning import loggers as pl_loggers
import pandas as pd
from pytorch_forecasting import (
    DeepAR,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import (
    MultivariateNormalDistributionLoss,
)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from typing import Dict
import numpy as np
import torch


def preprocess_df_dict(df_dict: Dict[str, pd.DataFrame]):
    data = []
    preprocess = lambda x: (100 * x["Close"].diff() / x["Close"].shift(1)).dropna()
    for key, df in df_dict.items():
        df = preprocess(df).to_frame()
        df["series"] = key
        df["date"] = df.index
        df["time_idx"] = range(df.shape[0])
        data.append(df)
    assert all([x.shape[0] == data[0].shape[0] for x in data[1:]])
    return pd.concat(data).reset_index(drop=True)


def get_training_cutoff(data, split_date):
    return data[data["date"] <= split_date]["time_idx"].max()


def get_dataloaders(
    data, training_cutoff, context_length, prediction_length, batch_size=32
):
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        static_categoricals=[
            "series"
        ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
        time_varying_unknown_reals=["Close"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=training_cutoff + 1
    )
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )
    return training, validation, train_dataloader, val_dataloader


def fit_model(
    training, train_dataloader, val_dataloader, log_dict="./multi_normal", max_epochs=30
):
    tensorboard = pl_loggers.TensorBoardLogger(log_dict)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=50,
        logger=tensorboard,
    )

    net = DeepAR.from_dataset(
        training,
        learning_rate=1e-2,
        log_interval=10,
        log_val_interval=1,
        hidden_size=30,
        rnn_layers=2,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=4),
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer


def date_timeidx_map(data):
    return data[["date", "time_idx"]].drop_duplicates().set_index("date")["time_idx"]


def prepend_tensor_with_ones(tensor: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((tensor.shape[0], 1))
    return torch.cat((ones, tensor), 1)


def forecastDecoratorDeepAR(
    df_dict,
    split_date,
    log_dict="./multi_normal",
    horizon=20,
    simulations=1000,
    max_encoder_length=30,
    max_epochs=30,
):

    data = preprocess_df_dict(df_dict)
    date_timeidx = date_timeidx_map(data)
    max_prediction_length = horizon
    training_cutoff = get_training_cutoff(data, split_date)

    context_length = max_encoder_length
    prediction_length = max_prediction_length
    training, validation, train_dataloader, val_dataloader = get_dataloaders(
        data,
        training_cutoff,
        context_length,
        prediction_length - 1,
    )
    trainer = fit_model(
        training,
        train_dataloader,
        val_dataloader,
        log_dict=log_dict,
        max_epochs=max_epochs,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = DeepAR.load_from_checkpoint(best_model_path)
    simulation = best_model.predict(
        val_dataloader,
        return_index=True,
        mode="samples",
        n_samples=simulations,
        trainer_kwargs=dict(accelerator="cpu"),
    )

    def forecast(
        date: str, asset, processLength=horizon, monteCarloTrials=simulations, **params
    ):
        assert processLength == horizon
        assert monteCarloTrials == simulations
        time_idx = date_timeidx[date]
        mask = np.logical_and(
            simulation.index["time_idx"] == time_idx,
            simulation.index["series"] == asset,
        )
        price = df_dict[asset]["Close"][df_dict[asset].index == date].values[0]
        output: torch.Tensor = simulation.output[mask].squeeze(0).T
        output = (1 + output) / 100
        output = output.cumprod(axis=1)
        output = prepend_tensor_with_ones(output)
        output = output * price
        return output

    return forecast
