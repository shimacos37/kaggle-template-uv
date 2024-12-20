import os
import sys

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
    test_df = pl.read_parquet(config.train.data.test_path)
    if config.train.data.do_standarize:
        df = pl.read_parquet(config.train.data.train_path)
        std = StandardScaler()
        numerical_cols = [
            col
            for col in df.columns
            if col not in config.train.data.remove_cols + config.train.data.cat_cols
        ]
        for col in numerical_cols:
            mean_ = df[col].mean()
            df = df.with_columns(pl.col(col).fill_null(mean_))
            test_df = test_df.with_columns(pl.col(col).fill_null(mean_))
            std.fit(df[[col]].to_numpy())
            test_df = test_df.with_columns(
                pl.Series(col, std.transform(test_df[[col]].to_numpy()).reshape(-1))
            )
    test_dataset = getattr(datasets, config.train.data.dataset_class)(
        test_df, config.train.data, mode=DatasetMode.TEST
    )
    config.train.model.num_feature = len(test_dataset[0]["numerical_features"])
    cpu_count = os.cpu_count()
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.data.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cpu_count if cpu_count is not None else 0,
    )
    model = Runner(config.train, num_training_steps=-1)
    # configの設定からtrainerを生成
    trainer: pytorch_lightning.Trainer = instantiate(config.train.trainer)
    preds = np.zeros([len(test_df), 6, 3])
    for n_fold in range(config.preprocess.n_splits):
        config.train.data.n_fold = n_fold
        outputs = trainer.predict(
            model,
            dataloaders=test_loader,
            ckpt_path=f"{config.train.output_dir}/{config.train.checkpoint_callback.filename}.ckpt",
        )
        preds += np.concatenate([x["preds"] for x in outputs]) / config.preprocess.n_splits
    ids = np.concatenate([x["ID"] for x in outputs])
    df = pl.DataFrame({"ID": ids, "preds": preds})
    df = (
        df.with_columns(pl.lit(["0", "1", "2", "3", "4", "5"]).alias("t"))
        .explode(["preds", "t"])
        .with_columns(pl.lit(["x", "y", "z"]).alias("loc"))
        .explode(["preds", "loc"])
        .with_columns((pl.col("loc") + "_" + pl.col("t")).alias("loc_t"))
        .pivot(index="ID", on="loc_t", values=["preds"])
    )
    df.write_parquet(f"{config.train.output_dir}/../test_result.parquet")


if __name__ == "__main__":
    main()
