import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

import hydra
import numpy as np
import polars as pl
import pytorch_lightning
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append(".")
from components.train.factories import datasets
from components.train.factories.datasets import DatasetMode
from components.train.runners import Runner


@hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    pytorch_lightning.seed_everything(seed=config.train.seed)
    df = pl.read_parquet(config.train.data.train_path)
    if config.train.data.do_standarize:
        std = StandardScaler()
        numerical_cols = [
            col
            for col in df.columns
            if col not in config.train.data.remove_cols + config.train.data.cat_cols
        ]
        for col in numerical_cols:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).mean()))
            df = df.with_columns(
                pl.Series(col, std.fit_transform(df[[col]].to_numpy()).reshape(-1))
            )
    train_df = df.filter(pl.col("fold") != config.train.data.n_fold)
    valid_df = df.filter(pl.col("fold") == config.train.data.n_fold)
    train_dataset = getattr(datasets, config.train.data.dataset_class)(
        train_df, config.train.data, mode=DatasetMode.TRAIN
    )
    valid_dataset = getattr(datasets, config.train.data.dataset_class)(
        valid_df, config.train.data, mode=DatasetMode.VALID
    )
    if len(train_dataset[0]["numerical_features"].shape) == 1:
        config.train.model.num_feature = train_dataset[0]["numerical_features"].shape[0]
    else:
        config.train.model.num_feature = train_dataset[0]["numerical_features"].shape[1]
    cpu_count = os.cpu_count()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.data.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count if cpu_count is not None else 0,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.data.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cpu_count if cpu_count is not None else 0,
    )

    num_training_steps = int(
        config.train.trainer.max_epochs
        * len(train_loader)
        / config.train.trainer.accumulate_grad_batches
    )

    model = Runner(config.train, num_training_steps=num_training_steps)
    # configの設定からtrainerを生成
    trainer = instantiate(config.train.trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if config.train.data.n_fold == config.preprocess.n_splits - 1:
        valid_df = pl.concat(
            [
                pl.read_parquet(f"{config.train.base_soutput_dir}/fold{i}/pred_results.parquet")
                for i in range(5)
            ]
        )
        valid_df = (
            valid_df.with_columns(pl.lit(["0", "1", "2", "3", "4", "5"]).alias("t"))
            .explode(["preds", "labels", "t"])
            .with_columns(pl.lit(["x", "y", "z"]).alias("loc"))
            .explode(["preds", "labels", "loc"])
            .with_columns((pl.col("loc") + "_" + pl.col("t")).alias("loc_t"))
            .pivot(index="ID", on="loc_t", values=["labels", "preds"])
            .sort("ID")
        )
        labels = valid_df.select(r"^labels_(x|y|z)_\d$").to_numpy()
        preds1 = valid_df.select(r"^preds_(x|y|z)_\d$").to_numpy()
        abs_diff = np.abs(labels - preds1)
        mae = np.mean(abs_diff.reshape(-1))
        print("MAE:", mae)
        valid_df.write_parquet(f"{config.train.base_soutput_dir}/valid_result.parquet")


if __name__ == "__main__":
    main()
