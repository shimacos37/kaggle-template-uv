import os
import re

import hydra
import numpy as np
import polars as pl
import polars.selectors as cs
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold


class Preprocessor:
    def __init__(self) -> None:
        self.gearShifter_mapping = {
            "drive": 0,
            "park": 1,
            "reverse": 2,
            "neutral": 3,
        }

    def run(self, df: pl.DataFrame, test_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = df.with_columns(
            pl.col("gearShifter").map_elements(
                lambda x: self.gearShifter_mapping[x], return_dtype=pl.Int32
            ),
            cs.by_dtype([pl.Boolean]).cast(float),
        )
        test_df = test_df.with_columns(
            pl.col("gearShifter").map_elements(
                lambda x: self.gearShifter_mapping[x], return_dtype=pl.Int32
            ),
            cs.by_dtype([pl.Boolean]).cast(float),
        )
        return df, test_df


@hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    config.preprocess.version = re.search(r"preprocess/(.+).py", __file__).group(1)
    os.makedirs(config.preprocess.output_dir, exist_ok=True)
    preprocessor = Preprocessor()
    df = pl.read_csv("./input/train_features.csv")
    test_df = pl.read_csv("./input/test_features.csv")
    df, test_df = preprocessor.run(df, test_df)
    label_df = (
        df.select(pl.col("ID"), pl.col(r"^(x|y|z)_\d$"))
        .unpivot(index=["ID"], variable_name="axis", value_name="loc")
        .with_columns(
            pl.col("axis").str.extract(r"(x|y|z)").alias("axis"),
            pl.col("axis").str.extract(r"(\d)").alias("t").cast(int),
            pl.col("ID").str.split("_").list[0].alias("scene_id"),
            (
                pl.col("ID").str.split("_").list[1].cast(int) / 10
                + pl.col("axis").str.extract(r"(\d)").cast(int) / 2
            ).alias("scene_second"),
        )
        .pivot(index=["scene_id", "scene_second", "ID", "t"], on=["axis"], values=["loc"])
        .sort(["scene_id", "scene_second", "t"])
    )
    trajectory_df = label_df.group_by(["ID", "scene_id"], maintain_order=True).agg(
        pl.concat_list(pl.col(["x", "y", "z"])).alias("trajectory")
    )
    df = df.drop(pl.col(r"^(x|y|z)_\d$")).join(trajectory_df, on="ID", how="left")
    gkfold = GroupKFold(n_splits=config.preprocess.n_splits)
    folds = np.zeros(len(df))
    for fold, (_, valid_idx) in enumerate(gkfold.split(df, groups=df["scene_id"])):
        folds[valid_idx] = fold
    df = df.with_columns(pl.Series("fold", folds))
    df.write_parquet(f"{config.preprocess.output_dir}/train.parquet")
    test_df.write_parquet(f"{config.preprocess.output_dir}/test.parquet")


if __name__ == "__main__":
    main()
