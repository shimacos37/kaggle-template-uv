import logging
import os
import pickle
from typing import Dict

import catboost as cat
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

plt.style.use("seaborn-v0_8-whitegrid")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)


plt.style.use("seaborn-v0_8-whitegrid")


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        feature_cols: list[str],
        cat_cols: list[str],
        label_col: str,
        pred_col: str,
        params: dict[str, int | float | str | list[int]],
        fix_label_bias: bool = False,
        negative_sampling: bool = False,
    ):
        self.model_dicts: dict[int, lgb.Booster] = {}
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.pred_col = pred_col
        self.params = params
        self.negative_sampling = negative_sampling
        self.fix_label_bias = fix_label_bias

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(self, train_df: pl.DataFrame) -> pl.DataFrame:
        importances = []
        oof_preds = np.zeros(len(train_df))
        train_df = train_df.with_columns(pl.Series("row_id", range(len(train_df))))
        for n_fold in range(5):
            bst = self.fit(train_df, n_fold)
            valid_df = train_df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = bst.predict(valid_df[self.feature_cols].to_numpy())
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance(importance_type="gain"))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        train_df = train_df.with_columns(pl.Series(self.pred_col, oof_preds))
        return train_df

    def fit(self, df: pl.DataFrame, n_fold: int) -> lgb.Booster:
        params = dict(self.params)
        train_df = df.filter((pl.col("fold") != n_fold) & (pl.col(self.label_col).is_not_null()))
        valid_df = df.filter(pl.col("fold") == n_fold)
        if self.negative_sampling:
            pos_df = train_df.filter(pl.col(self.label_col) == 1)
            neg_df = train_df.filter(pl.col(self.label_col) != 1).sample(fraction=0.1)
            train_df = pl.concat([pos_df, neg_df])
        if self.fix_label_bias:
            train_df.with_columns(pl.col(self.label_col) - pl.col("y_0"))
        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.label_col].to_numpy()

        X_valid = valid_df[self.feature_cols].to_numpy()
        y_valid = valid_df[self.label_col].to_numpy()
        print(
            f"{self.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            feature_name=self.feature_cols,
            categorical_feature=self.cat_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            feature_name=self.feature_cols,
            categorical_feature=self.cat_cols,
        )
        if params["objective"] == "lambdarank":
            params["label_gain"] = list(range(int(df[self.label_col].max() + 1)))
            train_group = (
                train_df.group_by(["original_file_id"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            valid_group = (
                valid_df.group_by(["original_file_id"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            lgtrain.set_group(train_group)
            lgvalid.set_group(valid_group)
            params["ndcg_eval_at"] = [5, 10]
        bst = lgb.train(
            params,
            lgtrain,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(200, first_metric_only=True),
                lgb.log_evaluation(1000),
            ],
        )
        print(
            f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
        )
        return bst

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        preds = np.zeros(len(df))
        for bst in self.model_dicts.values():
            preds += bst.predict(df[self.feature_cols].to_numpy())
        preds /= len(self.model_dicts)
        df = df.with_columns(pl.Series(self.pred_col, preds))
        return df

    def predict_oof(self, df: pl.DataFrame) -> pl.DataFrame:
        oof_preds = np.zeros(len(df))
        df = df.with_columns(pl.Series("row_id", range(len(df))))
        for n_fold in range(5):
            valid_df = df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = self.model_dicts[n_fold].predict(
                valid_df[self.feature_cols].to_numpy(),
            )
        df = df.with_columns(pl.Series(self.pred_col, oof_preds))
        return df

    def load_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/model{suffix}.pkl", "rb") as f:
            self.model_dicts = pickle.load(f)

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(os.path.join(model_dir, f"model{suffix}.pkl"), "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(self, result_dir: str, suffix: str = "") -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"importance{suffix}.png"))
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(result_dir, f"importance{suffix}.csv"),
            index=False,
        )


class XGBModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        feature_cols: list[str],
        cat_cols: list[str],
        label_col: str,
        pred_col: str,
        params: dict[str, int | float | str | list[int]],
        fix_label_bias: bool = False,
        negative_sampling: bool = False,
    ):
        self.model_dicts: dict[int, lgb.Booster] = {}
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.pred_col = pred_col
        self.params = params
        self.negative_sampling = negative_sampling
        self.fix_label_bias = fix_label_bias

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(self, train_df: pl.DataFrame) -> pl.DataFrame:
        importances = []
        oof_preds = np.zeros(len(train_df))
        train_df = train_df.with_columns(pl.Series("row_id", range(len(train_df))))
        for n_fold in range(5):
            bst = self.fit(train_df, n_fold)
            valid_df = train_df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = bst.predict(
                xgb.DMatrix(valid_df[self.feature_cols].to_numpy(), feature_names=self.feature_cols)
            )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_score(importance_type="gain")
            importances.append(
                [importance_dict[col] if col in importance_dict else 0 for col in self.feature_cols]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        train_df = train_df.with_columns(pl.Series(self.pred_col, oof_preds))
        return train_df

    def fit(self, df: pl.DataFrame, n_fold: int) -> xgb.Booster:
        params = dict(self.params)
        train_df = df.filter((pl.col("fold") != n_fold) & (pl.col(self.label_col).is_not_null()))
        valid_df = df.filter(pl.col("fold") == n_fold)
        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.label_col].to_numpy()

        X_valid = valid_df[self.feature_cols].to_numpy()
        y_valid = valid_df[self.label_col].to_numpy()
        print(
            f"{self.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self.feature_cols,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            label=y_valid,
            feature_names=self.feature_cols,
        )
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=100000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=200,
            # feval=self.custom_metric,
        )
        return bst

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        preds = np.zeros(len(df))
        for bst in self.model_dicts.values():
            preds += bst.predict(
                xgb.DMatrix(df[self.feature_cols].to_numpy(), feature_names=self.feature_cols)
            )
        preds /= len(self.model_dicts)
        df = df.with_columns(pl.Series(self.pred_col, preds))
        return df

    def predict_oof(self, df: pl.DataFrame) -> pl.DataFrame:
        oof_preds = np.zeros(len(df))
        df = df.with_columns(pl.Series("row_id", range(len(df))))
        for n_fold in range(5):
            valid_df = df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = self.model_dicts[n_fold].predict(
                xgb.DMatrix(
                    valid_df[self.feature_cols].to_pandas(), feature_names=self.feature_cols
                )
            )
        df = df.with_columns(pl.Series(self.pred_col, oof_preds))
        return df

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/model{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def load_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/model{suffix}.pkl", "rb") as f:
            self.model_dicts = pickle.load(f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.label_col + suffix}.csv",
            ),
            index=False,
        )


class CatModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        feature_cols: list[str],
        cat_cols: list[str],
        label_col: str,
        pred_col: str,
        params: dict[str, int | float | str | list[int]],
        use_calib_label: bool = False,
        use_pseudo_label: bool = False,
    ):
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.pred_col = pred_col
        self.params = params
        self.use_calib_label = use_calib_label
        self.use_pseudo_label = use_pseudo_label
        self.model_dicts: Dict[int, cat.Booster] = {}

    def store_model(self, bst: cat.CatBoost, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(self, train_df: pl.DataFrame, pseudo_df: pl.DataFrame | None = None) -> pl.DataFrame:
        importances = []
        oof_preds = np.zeros(len(train_df))
        train_df = train_df.with_columns(pl.Series("row_id", range(len(train_df))))
        for n_fold in range(5):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid_df = train_df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = bst.predict(
                cat.Pool(
                    valid_df[self.feature_cols].to_pandas(),
                    cat_features=self.cat_cols,
                )
            )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_feature_importance()
            importances.append(
                [importance_dict[col] if col in importance_dict else 0 for col in self.feature_cols]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        train_df = train_df.with_columns(pl.Series(self.pred_col, oof_preds))

        return train_df

    def fit(
        self, df: pl.DataFrame, n_fold: int, pseudo_df: pl.DataFrame | None = None
    ) -> cat.CatBoost:
        params = dict(self.params)
        train_df = df.filter((pl.col("fold") != n_fold) & (pl.col(self.label_col).is_not_null()))
        valid_df = df.filter(pl.col("fold") == n_fold)
        if self.use_pseudo_label and pseudo_df is not None:
            X_train = pd.concat(
                [train_df[self.feature_cols].to_pandas(), pseudo_df[self.feature_cols].to_pandas()]
            )
            if self.use_calib_label:
                y_train = pd.concat(
                    [
                        train_df[f"{self.label_col}_calib"].to_pandas(),
                        pseudo_df[self.label_col].to_pandas(),
                    ]
                )
            else:
                y_train = pd.concat(
                    [train_df[self.label_col].to_pandas(), pseudo_df[self.label_col].to_pandas()]
                )
        else:
            X_train = train_df[self.feature_cols].to_pandas()
            if self.use_calib_label:
                y_train = train_df[f"{self.label_col}_calib"].to_pandas()
            else:
                y_train = train_df[self.label_col].to_pandas()

        X_valid = valid_df[self.feature_cols].to_pandas()
        y_valid = valid_df[self.label_col].to_pandas()
        print(
            f"{self.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        dtrain = cat.Pool(
            X_train,
            label=y_train,
            feature_names=self.feature_cols,
            cat_features=self.cat_cols,
        )
        dvalid = cat.Pool(
            X_valid,
            label=y_valid,
            feature_names=self.feature_cols,
            cat_features=self.cat_cols,
        )
        bst = cat.train(
            pool=dtrain,
            params=params,
            num_boost_round=50000,
            evals=dvalid,
            early_stopping_rounds=100,
            verbose_eval=100,
            # feval=self.custom_metric,
        )
        return bst

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        preds = np.zeros(len(df))
        for bst in self.model_dicts.values():
            preds += bst.predict(
                cat.Pool(df[self.feature_cols].to_pandas(), cat_features=self.cat_cols)
            )
        preds /= len(self.model_dicts)
        df = df.with_columns(pl.Series(self.pred_col, preds))
        return df

    def predict_oof(self, df: pl.DataFrame) -> pl.DataFrame:
        oof_preds = np.zeros(len(df))
        df = df.with_columns(pl.Series("row_id", range(len(df))))
        for n_fold in range(5):
            valid_df = df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            oof_preds[row_idxs] = self.model_dicts[n_fold].predict(
                cat.Pool(
                    valid_df[self.feature_cols].to_pandas(),
                    cat_features=self.cat_cols,
                )
            )
        df = df.with_columns(pl.Series(self.pred_col, oof_preds))
        return df

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/model{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def load_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/model{suffix}.pkl", "rb") as f:
            self.model_dicts = pickle.load(f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.label_col + suffix}.csv",
            ),
            index=False,
        )
