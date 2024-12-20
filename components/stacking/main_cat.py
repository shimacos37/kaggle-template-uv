import logging
import os
import random
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error

sys.path.append(".")
from components.gbdt_train.models import CatModel
from components.stacking.utils import (
    load_gbdt_data,
    load_gbdt_test_data,
    load_nn_data,
    load_nn_test_data,
)

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="../../", config_name="config.yaml")
def main(config: DictConfig) -> None:
    set_seed(config.gbdt.seed)
    os.makedirs(config.stacking.output_dir, exist_ok=True)
    train_df = pl.read_parquet(config.stacking.data.train_path)
    test_df = pl.read_parquet(config.stacking.data.test_path)
    gbdt_df = load_gbdt_data(config.stacking.models.gbdt)
    nn_df = load_nn_data(config.stacking.models.nn)
    gbdt_test_df = load_gbdt_test_data(config.stacking.models.gbdt)
    nn_test_df = load_nn_test_data(config.stacking.models.nn)
    train_df = pl.concat([train_df, gbdt_df, nn_df], how="align")
    test_df = pl.concat([test_df, gbdt_test_df, nn_test_df], how="align")
    if config.stacking.use_pseudo_label:
        test_pred_df = pl.read_parquet(
            f"./output/stacking/{config.stacking.pseudo_model}/test_result.parquet"
        )
        test_df = test_df.join(
            test_pred_df.select(
                pl.col("ID"),
                pl.col(r"^(x|y|z)_\d_pred$")
                .cast(pl.Float64)
                .name.map(lambda x: x.rsplit("_pred")[0]),
            ),
            on="ID",
            how="left",
        )
    remove_cols = config.stacking.remove_cols
    maes = []
    for t in [5, 4, 3, 2, 1, 0]:
        for loc in ["x", "y", "z"]:
            match loc:
                case "x":
                    index = 0
                case "y":
                    index = 1
                case "z":
                    index = 2
            if config.stacking.use_calib_label:
                train_df = train_df.with_columns(
                    pl.col("trajectory_calib").list[t].list[index].alias(f"{loc}_{t}_calib")
                )
            train_df = train_df.with_columns(
                pl.col("trajectory").list[t].list[index].alias(f"{loc}_{t}")
            )
            if config.stacking.use_only_preds:
                feature_cols = [col for col in test_df.columns if col.endswith("_pred")]
                cat_cols = []
            else:
                feature_cols = [col for col in test_df.columns if col not in remove_cols]
                cat_cols = ["gearShifter"]
            if config.stacking.data.use_pred_feature:
                feature_cols += [f"{loc}_{i}_pred" for i in range(t) for loc in ["x", "y", "z"]]
            label_col = f"{loc}_{t}"
            pred_col = f"{loc}_{t}_pred"
            model = CatModel(
                feature_cols,
                cat_cols,
                label_col,
                pred_col,
                config.gbdt.catboost.params,
                use_calib_label=config.stacking.use_calib_label,
                use_pseudo_label=config.stacking.use_pseudo_label,
            )
            if os.path.exists(f"{config.stacking.output_dir}/model_{loc}_{t}.pkl"):
                model.load_model(config.stacking.output_dir, suffix=f"_{loc}_{t}")
                train_df = model.predict_oof(train_df)
                test_df = model.predict(test_df)
            else:
                train_df = model.cv(train_df, test_df)
                test_df = model.predict(test_df)
                model.save_model(config.stacking.output_dir, suffix=f"_{loc}_{t}")
                model.save_importance(config.stacking.output_dir, suffix=f"_{loc}_{t}")
            mae = mean_absolute_error(train_df[label_col].to_numpy(), train_df[pred_col].to_numpy())
            logger.info(f"fold all, MAE[{label_col}]: {mae}")
            maes.append(mae)
            remove_cols += [pred_col]
    train_df.write_parquet(f"{config.stacking.output_dir}/valid_result.parquet")
    test_df.write_parquet(f"{config.stacking.output_dir}/test_result.parquet")
    preds = train_df.select(pl.col(r"^(x|y|z)_\d_pred$")).to_numpy()
    labels = train_df.select(pl.col(r"^(x|y|z)_\d$")).to_numpy()
    abs_diff = np.abs(preds - labels)  # 各予測の差分の絶対値を計算して
    mae = np.mean(abs_diff.reshape(-1))
    logger.info(f"fold all, MAE: {mae}")
    _test_df = pl.read_csv("./input/test_features.csv")
    test_df = _test_df[["ID"]].join(test_df, how="left", on="ID")
    test_df.select(pl.col(r"^(x|y|z)_\d_pred$").name.map(lambda x: x.rsplit("_pred")[0])).write_csv(
        f"{config.stacking.output_dir}/submission.csv"
    )


if __name__ == "__main__":
    main()
