import polars as pl


def load_gbdt_data(model_names: list[str]) -> pl.DataFrame:
    pred_df = pl.concat(
        [
            pl.read_parquet(f"./output/gbdt/{model_name}/valid_result.parquet").select(
                pl.col("ID"), pl.col(r"^(x|y|z)_\d_pred$").name.prefix(f"{model_name}_")
            )
            for model_name in model_names
        ],
        how="align",
    )
    return pred_df


def load_nn_data(model_names: list[str]) -> pl.DataFrame:
    pred_df = pl.concat(
        [
            pl.read_parquet(f"./output/train/{model_name}/valid_result.parquet").select(
                pl.col("ID"),
                pl.col(r"^preds_(x|y|z)_\d$").name.map(
                    lambda x: f'{model_name}_{x.lstrip("preds_")}_pred'
                ),
            )
            for model_name in model_names
        ],
        how="align",
    )
    return pred_df


def load_gbdt_test_data(model_names: list[str]) -> pl.DataFrame:
    pred_df = pl.concat(
        [
            pl.read_parquet(f"./output/gbdt/{model_name}/test_result.parquet").select(
                pl.col("ID"), pl.col(r"^(x|y|z)_\d_pred$").name.prefix(f"{model_name}_")
            )
            for model_name in model_names
        ],
        how="align",
    )
    return pred_df


def load_nn_test_data(model_names: list[str]) -> pl.DataFrame:
    pred_df = pl.concat(
        [
            pl.read_parquet(f"./output/train/{model_name}/test_result.parquet").select(
                pl.col("ID"),
                pl.col(r"^(x|y|z)_\d$").name.map(lambda x: f"{model_name}_{x}_pred"),
            )
            for model_name in model_names
        ],
        how="align",
    )
    return pred_df
