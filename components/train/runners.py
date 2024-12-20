from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning
from omegaconf import DictConfig
from schedulefree import RAdamScheduleFree
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from components.train.factories import losses, models


class Runner(pytorch_lightning.LightningModule):
    def __init__(self, config: DictConfig, num_training_steps: int) -> None:
        super(Runner, self).__init__()
        self.config = config
        self.model = getattr(models, config.model.model_class)(config.model)
        self.num_training_steps = num_training_steps
        self.loss_fn = getattr(losses, config.loss.loss_class)(config.loss)
        self.best_score = float("inf")
        self.validation_step_outputs = []

    def configure_optimizers(self) -> Any:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.config.base.use_scheduler_free:
            optimizer = RAdamScheduleFree(optimizer_parameters, lr=self.config.base.lr)
        else:
            optimizer = AdamW(optimizer_parameters, lr=self.config.base.lr)
        optimizers = [optimizer]
        if self.config.base.use_scheduler:
            if self.config.base.scheduler == "linear_warmup":
                schedulers = [
                    {
                        "scheduler": get_linear_schedule_with_warmup(
                            optimizer=optimizer,
                            num_warmup_steps=int(self.num_training_steps * 0.05),
                            num_training_steps=self.num_training_steps,
                        ),  # type: ignore
                        "interval": "step",
                    },
                ]
            elif self.config.base.scheduler == "cosine_warmup":
                schedulers = [
                    {
                        "scheduler": get_cosine_schedule_with_warmup(
                            optimizer=optimizer,
                            num_warmup_steps=int(self.num_training_steps * 0.05),
                            num_training_steps=self.num_training_steps,
                        ),  # type: ignore
                        "interval": "step",
                    },
                ]

            return optimizers, schedulers
        else:
            return optimizers

    def forward(self, batch: Any) -> SequenceClassifierOutput:
        return self.model(batch)

    def on_train_start(self) -> None:
        if self.config.base.use_scheduler_free:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for optimizer in optimizers:
                    optimizer.train()
            else:
                optimizers.train()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        if self.config.model.use_scheduler_sampling:
            # sampling_probability = self.current_epoch / self.trainer.max_epochs
            if np.random.rand() < 0.5:
                # 普通の生成
                out = self.model.generate(batch)
            else:
                # teacher forcing
                out = self.model(batch)
        else:
            out = self.model.generate(batch)
        loss = self.loss_fn(out, batch["trajectory"].float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self) -> None:
        if self.config.base.use_scheduler_free:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for optimizer in optimizers:
                    optimizer.eval()
            else:
                optimizers.eval()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out = self.model.generate(batch)
        if len(out.shape) == 3:
            self.validation_step_outputs.append(
                {
                    "ID": batch["ID"],
                    "preds": out.detach().cpu().numpy(),
                    "labels": batch["trajectory"].detach().cpu().numpy(),
                }
            )
        else:
            # sceneで結合した時
            self.validation_step_outputs.append(
                {
                    "ID": np.array(batch["ID"]).T,  # list[str]で渡すと次元が[scene_len, bs]になる
                    "preds": out.detach().cpu().numpy(),
                    "labels": batch["trajectory"].detach().cpu().numpy(),
                }
            )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        out = self.model.generate(batch)
        output = {"ID": batch["ID"], "preds": out.detach().cpu().numpy()}
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        ids = np.concatenate([x["ID"] for x in outputs]).reshape(
            -1,
        )
        preds = np.concatenate([x["preds"] for x in outputs]).reshape(-1, 6, 3)
        labels = np.concatenate([x["labels"] for x in outputs]).reshape(-1, 6, 3)
        df = pl.DataFrame({"ID": ids, "preds": preds, "labels": labels})
        df = df.filter(pl.col("ID") != "pad")
        mask = np.where(labels != -100)
        abs_diff = np.abs(preds[mask] - labels[mask])  # 各予測の差分の絶対値を計算して
        mae = np.mean(abs_diff.reshape(-1))
        if len(df) >= 100 and mae < self.best_score:
            self.best_score = mae
            df.write_parquet(f"{self.config.output_dir}/pred_results.parquet")
        self.log("val_loss", mae, prog_bar=True)
        self.validation_step_outputs.clear()
