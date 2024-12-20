import os
import re

import hydra
import numpy as np
import polars as pl
import simdkalman
from omegaconf import DictConfig


def setup_kf(ob_noise=5e-5):
    T = 0.5
    # 状態遷移行列
    state_transition = np.array(
        [
            [1, 0, 0, T, 0, 0, 0.5 * T**2, 0, 0],
            [0, 1, 0, 0, T, 0, 0, 0.5 * T**2, 0],
            [0, 0, 1, 0, 0, T, 0, 0, 0.5 * T**2],
            [0, 0, 0, 1, 0, 0, T, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, T, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, T],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # プロセスノイズ
    process_noise = (
        np.diag([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 1e-6]) + np.ones((9, 9)) * 1e-9
    )

    # 観測モデル
    observation_model = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]
    )

    # 観測ノイズ
    observation_noise = np.diag([ob_noise, ob_noise, ob_noise]) + np.ones((3, 3)) * 1e-9

    # Kalmanフィルターの設定
    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise,
    )
    return kf


@hydra.main(config_path="../../", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    config.postprocess.version = re.search(r"postprocess/(.+).py", __file__).group(1)
    os.makedirs(config.postprocess.output_dir, exist_ok=True)
    pred_cols = [f"{c}_{t}_pred" for t in range(6) for c in ["x", "y", "z"]]
    label_cols = [f"{c}_{t}" for t in range(6) for c in ["x", "y", "z"]]
    MODEL_NAMES = [
        "exp062_stacking_catboost_pseudo_label",
        "exp054_stacking_lgbm",
        "exp068_stacking_catboost",
    ]
    preds = [
        pl.read_parquet(f"./output/stacking/{model_version}/valid_result.parquet")
        .sort("ID")
        .select(pred_cols)
        .to_numpy()
        for model_version in MODEL_NAMES
    ]
    labels = (
        pl.read_parquet(f"./output/stacking/{MODEL_NAMES[0]}/valid_result.parquet")
        .sort("ID")
        .select(label_cols)
        .to_numpy()
    )
    for pred, model_name in zip(preds, MODEL_NAMES, strict=False):
        abs_diff = np.abs(labels - pred)
        mae = np.mean(abs_diff.reshape(-1))
        print(f"MAE({model_name})", mae)
    preds = np.mean(preds, axis=0)
    abs_diff = np.abs(labels - preds)
    mae = np.mean(abs_diff.reshape(-1))
    print("MAE(平均)", mae)

    preds = preds.reshape(-1, 6, 3)
    labels = labels.reshape(-1, 6, 3)
    kf = setup_kf()
    smoothed = kf.smooth(preds)

    preds = smoothed.states.mean[:, :, :3]
    abs_diff = np.abs(labels - preds)
    mae = np.mean(abs_diff.reshape(-1))
    print("MAE(smooth)", mae)

    test_df = pl.read_csv("./input/test_features.csv")
    test_preds = [
        test_df[["ID"]]
        .join(
            pl.read_parquet(f"./output/stacking/{model_version}/test_result.parquet"),
            how="left",
            on="ID",
        )
        .select(pred_cols)
        .to_numpy()
        for model_version in MODEL_NAMES
    ]
    test_preds = np.mean(test_preds, axis=0)
    test_preds = test_preds.reshape(-1, 6, 3)
    kf = setup_kf()

    smoothed = kf.smooth(test_preds)
    smoothed_preds = smoothed.states.mean[:, :, :3].reshape(-1, 18)
    pl.DataFrame(smoothed_preds, schema=label_cols).write_csv(
        f"{config.postprocess.output_dir}/test_result_smoothing.csv"
    )


if __name__ == "__main__":
    main()
