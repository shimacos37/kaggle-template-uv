from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
import polars as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from timm.data import resolve_data_config
from transformers import CLIPImageProcessor


@dataclass
class DatasetOutput:
    image: torch.Tensor
    features: npt.NDArray[np.float64]
    trajectory: npt.NDArray[np.float64] | None = None


class DatasetMode(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


def load_image(id_: str) -> dict[str, npt.NDArray[np.uint8]]:
    images = {
        "image": np.array(Image.open(f"./input/images/{id_}/image_t.png").convert("RGB")),
        "image1": np.array(Image.open(f"./input/images/{id_}/image_t-0.5.png").convert("RGB")),
        "image2": np.array(Image.open(f"./input/images/{id_}/image_t-1.0.png").convert("RGB")),
    }
    return images


def load_mask(id_: str) -> dict[str, npt.NDArray[np.uint8]]:
    try:
        masks = {
            "mask": np.array(Image.open(f"./input/images/{id_}/mask_t.png").convert("RGB")),
            "mask1": np.array(Image.open(f"./input/images/{id_}/mask_t-0.5.png").convert("RGB")),
            "mask2": np.array(Image.open(f"./input/images/{id_}/mask_t-1.0.png").convert("RGB")),
        }
    except FileNotFoundError:
        masks = {
            "mask": np.zeros((64, 128, 3), dtype=np.uint8),
            "mask1": np.zeros((64, 128, 3), dtype=np.uint8),
            "mask2": np.zeros((64, 128, 3), dtype=np.uint8),
        }
    return masks


def load_depth(id_: str) -> dict[str, npt.NDArray[np.uint8]]:
    depths = {
        "depth": np.load(f"./input/images/{id_}/depth_t.npy"),
        "depth1": np.load(f"./input/images/{id_}/depth_t-0.5.npy"),
        "depth2": np.load(f"./input/images/{id_}/depth_t-1.0.npy"),
    }
    return depths


@lru_cache(maxsize=128)
def load_image_list(id_: str) -> list[npt.NDArray[np.uint8]]:
    images = [
        np.array(Image.open(f"./input/images/{id_}/image_t.png").convert("RGB")),
        np.array(Image.open(f"./input/images/{id_}/image_t-0.5.png").convert("RGB")),
        np.array(Image.open(f"./input/images/{id_}/image_t-1.0.png").convert("RGB")),
    ]
    return images


class MotionDataset:
    def __init__(self, df: pl.DataFrame, config: DictConfig, mode: DatasetMode):
        pretrained_cfg = timm.get_pretrained_cfg(config.image_model)
        self.data_config = resolve_data_config({}, pretrained_cfg=pretrained_cfg.to_dict())
        self.mode = mode
        self.config = config

        self.ids = df[config.id_col].to_list()
        numerical_cols = [
            col for col in df.columns if col not in config.remove_cols + config.cat_cols
        ]
        self.numerical_features = df[numerical_cols].fill_null(0).to_numpy()
        self.cat_features = df[config.cat_cols].to_dicts()
        if mode != DatasetMode.TEST:
            self.trajectory = df["trajectory"].to_list()

    def __len__(self) -> int:
        return len(self.ids)

    def _augmenation(self, p: float = 0.5) -> A.Compose:
        aug_list: list[Any] = []

        if self.mode == DatasetMode.TRAIN:
            aug_list.extend(
                [
                    # RandomCrop(height, width, p),
                    A.OneOf([A.RGBShift(), A.HueSaturationValue()], p),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.8),
                            A.RandomGamma(),
                        ],
                        p,
                    ),
                    A.OneOf([A.GridDropout(), A.CoarseDropout()], p),
                    # Downscale(scale_min=0.8, scale_max=0.99, p=p),
                ]
            )

        aug_list.extend(
            [
                A.Resize(
                    self.config.image_height,
                    self.config.image_width,
                    cv2.INTER_LINEAR,
                ),
                A.Normalize(
                    mean=self.data_config["mean"],
                    std=self.data_config["std"],
                    max_pixel_value=255,
                    p=1,
                ),
                ToTensorV2(),
            ]
        )
        return A.Compose(
            aug_list,
            additional_targets={
                "image0": "image",
                "image1": "image",
                "image2": "image",
                "mask1": "mask",
                "mask2": "mask",
                "depth": "mask",
                "depth1": "mask",
                "depth2": "mask",
            },
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        id_ = self.ids[idx]
        images = load_image(id_)
        if self.config.use_mask:
            masks = load_mask(id_)
            images.update(masks)
        if self.config.use_depth:
            depths = load_depth(id_)
            images.update(depths)
        numerical_features = self.numerical_features[idx]
        cat_features = self.cat_features[idx]
        transformed_images = self._augmenation()(**images)
        cat_image = torch.cat(
            [
                transformed_images["image"],
                transformed_images["image1"],
                transformed_images["image2"],
            ]
        )
        if self.config.use_mask and self.config.use_depth:
            cat_image = torch.cat(
                [
                    transformed_images["image"],
                    transformed_images["mask"].permute(2, 0, 1),
                    transformed_images["depth"].unsqueeze(0),
                    transformed_images["image1"],
                    transformed_images["mask1"].permute(2, 0, 1),
                    transformed_images["depth1"].unsqueeze(0),
                    transformed_images["image2"],
                    transformed_images["mask2"].permute(2, 0, 1),
                    transformed_images["depth2"].unsqueeze(0),
                ]
            )
        elif self.config.use_mask and not self.config.use_depth:
            cat_image = torch.cat(
                [
                    transformed_images["image"],
                    transformed_images["mask"].permute(2, 0, 1),
                    transformed_images["image1"],
                    transformed_images["mask1"].permute(2, 0, 1),
                    transformed_images["image2"],
                    transformed_images["mask2"].permute(2, 0, 1),
                ]
            )
        elif not self.config.use_mask and self.config.use_depth:
            cat_image = torch.cat(
                [
                    transformed_images["image"],
                    transformed_images["depth"].unsqueeze(0),
                    transformed_images["image1"],
                    transformed_images["depth1"].unsqueeze(0),
                    transformed_images["image2"],
                    transformed_images["depth2"].unsqueeze(0),
                ]
            )
        else:
            cat_image = torch.cat(
                [
                    transformed_images["image"],
                    transformed_images["image1"],
                    transformed_images["image2"],
                ]
            )
        out = {
            "ID": id_,
            "image": cat_image,
            "numerical_features": numerical_features,
        }
        for key, val in cat_features.items():
            out[key] = val

        if self.mode != DatasetMode.TEST:
            out["trajectory"] = np.array(self.trajectory[idx])
            return out
        else:
            return out


class MotionSceneDataset:
    def __init__(self, df: pl.DataFrame, config: DictConfig, mode: DatasetMode):
        df = df.sort(["scene_id", "scene_second"])
        pretrained_cfg = timm.get_pretrained_cfg(config.image_model)
        self.data_config = resolve_data_config({}, pretrained_cfg=pretrained_cfg.to_dict())
        self.mode = mode
        self.config = config

        self.ids = (
            df.group_by("scene_id", maintain_order=True)
            .agg(pl.col(config.id_col))[config.id_col]
            .to_list()
        )
        numerical_cols = [
            col for col in df.columns if col not in config.remove_cols + config.cat_cols
        ]
        self.numerical_features = (
            df.group_by("scene_id", maintain_order=True)
            .agg(pl.concat_list(numerical_cols).alias("numerical_features"))["numerical_features"]
            .to_list()
        )
        self.cat_features = (
            df.group_by("scene_id", maintain_order=True).agg(pl.col(config.cat_cols))
        )[config.cat_cols].to_dicts()
        if mode != DatasetMode.TEST:
            self.trajectory = (
                df.group_by("scene_id", maintain_order=True)
                .agg(pl.col("trajectory"))["trajectory"]
                .to_list()
            )
        self.max_len = 6
        self.additional_targets = {f"image{i}": "image" for i in range(18)}
        if config.use_mask:
            self.additional_targets.update({f"mask{i}": "mask" for i in range(18)})
        if config.use_depth:
            self.additional_targets.update({f"depth{i}": "mask" for i in range(18)})

    def __len__(self) -> int:
        return len(self.ids)

    def _augmenation(self, p: float = 0.5) -> A.Compose:
        aug_list: list[Any] = []

        aug_list.extend(
            [
                A.Resize(
                    self.config.image_height,
                    self.config.image_width,
                    cv2.INTER_LINEAR,
                ),
                A.Normalize(
                    mean=self.data_config["mean"],
                    std=self.data_config["std"],
                    max_pixel_value=255,
                    p=1,
                ),
                ToTensorV2(),
            ]
        )
        return A.Compose(
            aug_list,
            additional_targets=self.additional_targets,
        )

    def padding_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # 現在の1次元目のサイズを取得
        current_size = tensor.size(0)
        # 6次元目まで埋めるためのパディングを計算
        if current_size < self.max_len:
            padding_size = self.max_len - current_size
            # 0でパディング
            padding = torch.zeros((padding_size, *tensor.size()[1:]), dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding), dim=0)
        return tensor

    def padding_numpy(self, array: np.ndarray, value: int = 0) -> np.ndarray:
        # 現在の1次元目のサイズを取得
        current_size = array.shape[0]
        # 6次元目まで埋めるためのパディングを計算
        if current_size < self.max_len:
            padding_size = self.max_len - current_size
            # 0でパディング
            pad_width = [(0, padding_size)] + [(0, 0)] * (array.ndim - 1)
            array = np.pad(array, pad_width, mode="constant", constant_values=value)
        return array

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ids = self.ids[idx]
        images = [load_image(id_) for id_ in ids]
        transformed_images = [self._augmenation()(**image) for image in images]
        # if self.config.use_mask:
        #     masks = [load_mask(id_) for id_ in ids]
        #     transformed_masks = [self._augmenation()(**mask) for mask in masks]
        # if self.config.use_depth:
        #     depths = [load_depth(id_) for id_ in ids]
        #     transformed_depths = [self._augmenation()(**depth) for depth in depths]
        numerical_features = np.stack(self.numerical_features[idx])
        numerical_features = self.padding_numpy(numerical_features)

        cat_features = self.cat_features[idx]
        for key in cat_features.keys():
            cat_features[key] = self.padding_numpy(np.array(cat_features[key]))

        cat_image = torch.stack(
            [
                torch.cat(
                    [
                        transformed_image["image"],
                        transformed_image["image1"],
                        transformed_image["image2"],
                    ]
                )
                for transformed_image in transformed_images
            ]
        )
        cat_image = self.padding_tensor(cat_image)
        out = {
            "ID": [id_ for id_ in ids] + ["pad"] * (self.max_len - len(ids)),
            "image": cat_image,
            "numerical_features": numerical_features,
        }
        for key, val in cat_features.items():
            out[key] = val

        if self.mode != DatasetMode.TEST:
            trajectory = np.stack(self.trajectory[idx])
            trajectory = self.padding_numpy(trajectory, value=-100)
            out["trajectory"] = trajectory
            return out
        else:
            return out


class MotionTableDataset:
    def __init__(self, df: pl.DataFrame, config: DictConfig, mode: DatasetMode):
        pretrained_cfg = timm.get_pretrained_cfg(config.image_model)
        self.data_config = resolve_data_config({}, pretrained_cfg=pretrained_cfg.to_dict())
        self.mode = mode
        self.config = config

        self.ids = df[config.id_col].to_list()
        numerical_cols = [
            col for col in df.columns if col not in config.remove_cols + config.cat_cols
        ]
        self.numerical_features = df[numerical_cols].fill_null(0).to_numpy()
        self.cat_features = df[config.cat_cols].to_dicts()
        if mode != DatasetMode.TEST:
            self.trajectory = df["trajectory"].to_list()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        id_ = self.ids[idx]
        numerical_features = self.numerical_features[idx]
        cat_features = self.cat_features[idx]
        out = {
            "ID": id_,
            "numerical_features": numerical_features,
        }
        for key, val in cat_features.items():
            out[key] = val
        if self.mode != DatasetMode.TEST:
            out["trajectory"] = np.array(self.trajectory[idx])
            return out
        else:
            return out


class MotionCLIPDataset:
    def __init__(self, df: pl.DataFrame, config: DictConfig, mode: DatasetMode):
        self.preprocessor = CLIPImageProcessor.from_pretrained(config.image_model)
        self.mode = mode
        self.config = config

        self.ids = df[config.id_col].to_list()
        numerical_cols = [
            col for col in df.columns if col not in config.remove_cols + config.cat_cols
        ]
        self.numerical_features = df[numerical_cols].fill_null(0).to_numpy()
        self.cat_features = df[config.cat_cols].to_dicts()
        if mode != DatasetMode.TEST:
            self.trajectory = df["trajectory"].to_list()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        id_ = self.ids[idx]
        images = load_image_list(id_)
        numerical_features = self.numerical_features[idx]
        cat_features = self.cat_features[idx]
        transformed_images = self.preprocessor(images)
        out = {
            "ID": id_,
            "image": transformed_images["pixel_values"],
            "numerical_features": numerical_features,
        }
        for key, val in cat_features.items():
            out[key] = val
        if self.mode != DatasetMode.TEST:
            out["trajectory"] = np.array(self.trajectory[idx])
            return out
        else:
            return out
