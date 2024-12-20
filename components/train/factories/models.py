from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import CLIPModel


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    上三角行列を生成し、未来情報を遮断するcausal maskを作成
    """
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)  # 対角線より上を1
    mask = mask.masked_fill(mask == 1, float("-inf"))  # 無限小を代入
    return mask


@dataclass
class EmbeddingInfo:
    cat_name: str
    vocab_size: int
    embedding_dim: int


class CatEncoder(nn.Module):
    def __init__(self) -> None:
        super(CatEncoder, self).__init__()
        encoder_dict = {}
        encoder_infos = [
            EmbeddingInfo(cat_name="gearShifter", vocab_size=4, embedding_dim=32),
        ]
        for encoder_info in encoder_infos:
            encoder_dict[encoder_info.cat_name] = nn.Embedding(
                encoder_info.vocab_size, encoder_info.embedding_dim
            )
        self.encoder_dict = nn.ModuleDict(encoder_dict)
        self.out_features = sum([encoder_info.embedding_dim for encoder_info in encoder_infos])

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x_dict: [bs, seq_len, 1]
        return: [bs, seq_len, self.encoder.out_features]
        """
        outs = []
        for cat, val in x_dict.items():
            if cat in self.encoder_dict.keys():
                outs.append(self.encoder_dict[cat](val.long()))
        return torch.cat(outs, dim=-1)


class MotionTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionTransformer, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.encoder = nn.Sequential(
            nn.Linear(
                config.num_feature + self.image_model.num_features + self.cat_encoder.out_features,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.decoder_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
        # Transformerデコーダ
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size, nhead=config.decoder_num_heads, batch_first=True
            ),
            num_layers=config.decoder_num_layers,
        )
        self.seq_len = config.seq_len
        self.use_teacher_forcing = config.use_teacher_forcing
        self.trajectory_linear = nn.Linear(config.num_classes, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, config.num_classes)

    def encode(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = self.image_model(batch["image"])
        cat_embeddings = self.cat_encoder(batch)
        features = torch.cat(
            [image_features, batch["numerical_features"].half(), cat_embeddings], dim=-1
        )
        encoder_output = self.encoder(features)
        return encoder_output

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        if self.use_teacher_forcing:
            trajectry_features = self.trajectory_linear(batch["trajectory"].half())
            tgt = torch.cat([tgt_start, trajectry_features[:, :-1, :]], dim=1)
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=self.seq_len),
            )
        else:
            output_sequence = []
            for t in range(self.seq_len):
                tgt = (
                    torch.cat([tgt_start] + output_sequence, dim=1)
                    if output_sequence
                    else tgt_start
                )
                decoder_output = self.decoder(
                    tgt=tgt,
                    memory=encoder_output.unsqueeze(1),
                    tgt_mask=generate_causal_mask(seq_len=t + 1),
                )
                next_pred = decoder_output[:, -1:, :]  # 最後の時刻の出力
                output_sequence.append(next_pred)
            decoder_output = torch.cat(output_sequence, dim=1)
        output = self.output_linear(decoder_output)
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        decoder_inputs = []
        outputs = []
        for t in range(self.seq_len):
            tgt = torch.cat([tgt_start] + decoder_inputs, dim=1) if decoder_inputs else tgt_start
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=t + 1),
            )
            output = self.output_linear(decoder_output)[:, -1:, :]  # 最後の時刻の出力
            next_pred = self.trajectory_linear(output)
            outputs.append(output)
            decoder_inputs.append(next_pred)
        output = torch.cat(outputs, dim=1)
        return output


class MotionTransformerV2(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionTransformerV2, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.table_feature_linear = nn.Linear(
            config.num_feature + self.cat_encoder.out_features, config.hidden_size
        )
        self.encoder = nn.Sequential(
            nn.Linear(
                self.image_model.num_features + config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
        )
        self.decoder_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
        # Transformerデコーダ
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size, nhead=config.decoder_num_heads, batch_first=True
            ),
            num_layers=config.decoder_num_layers,
        )
        self.seq_len = config.seq_len
        self.use_teacher_forcing = config.use_teacher_forcing
        self.trajectory_linear = nn.Linear(config.num_classes, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size + config.hidden_size, config.num_classes)

    def encode(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = self.image_model(batch["image"])
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        return encoder_output, table_features

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output, table_features = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        if self.use_teacher_forcing:
            trajectry_features = self.trajectory_linear(batch["trajectory"].half())
            tgt = torch.cat([tgt_start, trajectry_features[:, :-1, :]], dim=1)
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=self.seq_len),
            )
        else:
            output_sequence = []
            for t in range(self.seq_len):
                tgt = (
                    torch.cat([tgt_start] + output_sequence, dim=1)
                    if output_sequence
                    else tgt_start
                )
                decoder_output = self.decoder(
                    tgt=tgt,
                    memory=encoder_output.unsqueeze(1),
                    tgt_mask=generate_causal_mask(seq_len=t + 1),
                )
                next_pred = decoder_output[:, -1:, :]  # 最後の時刻の出力
                output_sequence.append(next_pred)
            decoder_output = torch.cat(output_sequence, dim=1)
        output = self.output_linear(
            torch.cat(
                [decoder_output, table_features.unsqueeze(1).expand(-1, self.seq_len, -1)], dim=-1
            )
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output, table_features = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        decoder_inputs = []
        outputs = []
        for t in range(self.seq_len):
            tgt = torch.cat([tgt_start] + decoder_inputs, dim=1) if decoder_inputs else tgt_start
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=t + 1),
            )
            output = self.output_linear(
                torch.cat([decoder_output[:, -1, :], table_features], dim=-1)
            ).unsqueeze(1)  # 最後の時刻の出力
            next_pred = self.trajectory_linear(output)
            outputs.append(output)
            decoder_inputs.append(next_pred)
        output = torch.cat(outputs, dim=1)
        return output


class MotionTransformerV3(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionTransformerV3, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.image_feature_linear = nn.Sequential(
            nn.Linear(config.image_model, config.hidden_size),
            nn.ReLU(),
        )
        self.table_feature_linear = nn.Sequential(
            nn.Linear(config.num_feature + self.cat_encoder.out_features, config.hidden_size),
            nn.ReLU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.image_model.num_features,
                nhead=config.encoder_num_heads,
                batch_first=True,
            ),
            num_layers=config.encoder_num_layers,
        )
        self.decoder_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
        # Transformerデコーダ
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size, nhead=config.decoder_num_heads, batch_first=True
            ),
            num_layers=config.decoder_num_layers,
        )
        self.seq_len = config.seq_len
        self.use_teacher_forcing = config.use_teacher_forcing
        self.trajectory_linear = nn.Linear(config.num_classes, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size + config.hidden_size, config.num_classes)

    def encode(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features1 = self.image_feature_linear(self.image_model(batch["image"][:, :3, :, :]))
        image_features2 = self.image_feature_linear(self.image_model(batch["image"][:, 3:6, :, :]))
        image_features3 = self.image_feature_linear(self.image_model(batch["image"][:, 6:, :, :]))
        image_features = torch.cat(
            [
                image_features1.unsqueeze(1),
                image_features2.unsqueeze(1),
                image_features3.unsqueeze(1),
            ],
            dim=1,
        )
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        encoder_output = self.encoder(image_features)
        return encoder_output, table_features

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output, table_features = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        if self.use_teacher_forcing:
            trajectry_features = self.trajectory_linear(batch["trajectory"].half())
            tgt = torch.cat([tgt_start, trajectry_features[:, :-1, :]], dim=1)
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=self.seq_len),
            )
        else:
            output_sequence = []
            for t in range(self.seq_len):
                tgt = (
                    torch.cat([tgt_start] + output_sequence, dim=1)
                    if output_sequence
                    else tgt_start
                )
                decoder_output = self.decoder(
                    tgt=tgt,
                    memory=encoder_output.unsqueeze(1),
                    tgt_mask=generate_causal_mask(seq_len=t + 1),
                )
                next_pred = decoder_output[:, -1:, :]  # 最後の時刻の出力
                output_sequence.append(next_pred)
            decoder_output = torch.cat(output_sequence, dim=1)
        output = self.output_linear(
            torch.cat(
                [decoder_output, table_features.unsqueeze(1).expand(-1, self.seq_len, -1)], dim=-1
            )
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output, table_features = self.encode(batch)
        tgt_start = self.decoder_embedding.expand(encoder_output.size(0), -1, -1)
        decoder_inputs = []
        outputs = []
        for t in range(self.seq_len):
            tgt = torch.cat([tgt_start] + decoder_inputs, dim=1) if decoder_inputs else tgt_start
            decoder_output = self.decoder(
                tgt=tgt,
                memory=encoder_output.unsqueeze(1),
                tgt_mask=generate_causal_mask(seq_len=t + 1),
            )
            output = self.output_linear(
                torch.cat([decoder_output[:, -1, :], table_features], dim=-1)
            ).unsqueeze(1)  # 最後の時刻の出力
            next_pred = self.trajectory_linear(output)
            outputs.append(output)
            decoder_inputs.append(next_pred)
        output = torch.cat(outputs, dim=1)
        return output


class MotionMLP(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionMLP, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.encoder = nn.Sequential(
            nn.Linear(
                config.num_feature + self.image_model.num_features + self.cat_encoder.out_features,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(config.hidden_size, config.num_classes * self.seq_len)

    def encode(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = self.image_model(batch["image"])
        cat_embeddings = self.cat_encoder(batch)
        features = torch.cat(
            [image_features, batch["numerical_features"].half(), cat_embeddings], dim=-1
        )
        encoder_output = self.encoder(features)
        return encoder_output

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        encoder_output = self.encode(batch)
        output = self.output_linear(encoder_output).view(-1, self.seq_len, self.num_classes)
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionMLPV2(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionMLPV2, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.table_feature_linear = nn.Linear(
            config.num_feature + self.cat_encoder.out_features, config.hidden_size
        )
        self.encoder = nn.Sequential(
            nn.Linear(
                self.image_model.num_features + config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(
            config.hidden_size + config.hidden_size, config.num_classes * self.seq_len
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = self.image_model(batch["image"])
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionLSTM(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionLSTM, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.table_feature_linear = nn.Linear(
            config.num_feature + self.cat_encoder.out_features, config.hidden_size
        )
        self.lstm = nn.LSTM(
            self.image_model.num_features + config.hidden_size,
            config.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.seq_len = config.seq_len
        self.in_channels = config.in_channels
        self.scene_len = 6
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(
            config.hidden_size * 2 + config.hidden_size, config.num_classes * self.seq_len
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        bs, scene_len, _, h, w = batch["image"].shape
        image_features = self.image_model(batch["image"].view(-1, self.in_channels, h, w))
        image_features = image_features.view(bs, scene_len, -1)
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output, _ = self.lstm(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, scene_len, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class Conv3dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
    ):
        super().__init__(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        )


class Motion3DMLPV2(nn.Module):
    def __init__(self, config: DictConfig):
        super(Motion3DMLPV2, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.table_feature_linear = nn.Linear(
            config.num_feature + self.cat_encoder.out_features, config.hidden_size
        )
        k = 2
        self.conv3d = torch.nn.Sequential(
            Conv3dBlock(
                self.image_model.num_features,
                self.image_model.num_features,
                (2, 2, k),
                (0, k // 2, k // 2),
            ),
            Conv3dBlock(
                self.image_model.num_features,
                self.image_model.num_features,
                (2, k, k),
                (0, k // 2, k // 2),
            ),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.encoder = nn.Sequential(
            nn.Linear(
                self.image_model.num_features + config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.in_channels = config.in_channels
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.n_frames = 3
        self.output_linear = nn.Linear(
            config.hidden_size + config.hidden_size, config.num_classes * self.seq_len
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        image_feature_map1 = self.image_model.forward_features(
            batch["image"][:, : self.in_channels, :, :]
        )
        image_feature_map2 = self.image_model.forward_features(
            batch["image"][:, self.in_channels : self.in_channels * 2, :, :]
        )
        image_feature_map3 = self.image_model.forward_features(
            batch["image"][:, self.in_channels * 2 :, :, :]
        )
        image_feature_map = torch.cat(
            [
                image_feature_map1.unsqueeze(1),
                image_feature_map2.unsqueeze(1),
                image_feature_map3.unsqueeze(1),
            ],
            dim=1,
        ).transpose(1, 2)
        image_features = self.conv3d(image_feature_map).squeeze(2, 3, 4)
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionCLIPMLPV2(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionCLIPMLPV2, self).__init__()
        self.image_model = CLIPModel.from_pretrained(config.image_model)
        self.cat_encoder = CatEncoder()
        self.table_feature_linear = nn.Linear(
            config.num_feature + self.cat_encoder.out_features, config.hidden_size
        )
        self.encoder = nn.Sequential(
            nn.Linear(
                self.image_model.projection_dim * 2 + config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(
            config.hidden_size + config.hidden_size, config.num_classes * self.seq_len
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = torch.stack(
            [self.image_model.get_image_features(image) for image in batch["image"]]
        )
        _mean = image_features.mean(0)
        _max = image_features.max(0).values
        image_features = torch.cat([_mean, _max], dim=-1)
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MLPV2(nn.Module):
    def __init__(self, config: DictConfig):
        super(MLPV2, self).__init__()
        self.cat_encoder = CatEncoder()
        self.encoder = nn.Sequential(
            nn.Linear(
                config.num_feature + self.cat_encoder.out_features,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(
            config.hidden_size + config.num_feature + self.cat_encoder.out_features,
            config.num_classes * self.seq_len,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        cat_embeddings = self.cat_encoder(batch)
        table_features = torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        encoder_output = self.encoder(table_features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionDiffImageMLPV2(MotionMLPV2):
    def __init__(self, config: DictConfig):
        super(MotionDiffImageMLPV2, self).__init__(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        diff_images = torch.cat(
            [
                batch["image"][:, :3, :, :] - batch["image"][:, 3:6, :, :],
                batch["image"][:, 3:6, :, :] - batch["image"][:, 6:, :, :],
            ],
            dim=1,
        )
        image_features = self.image_model(diff_images)
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionDiffImageMLPV3(MotionMLPV2):
    def __init__(self, config: DictConfig):
        super(MotionDiffImageMLPV3, self).__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(
                self.image_model.num_features * 2 + config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features1 = self.image_model(batch["image"][:, :3, :, :])
        image_features2 = self.image_model(batch["image"][:, 3:6, :, :])
        image_features3 = self.image_model(batch["image"][:, 6:, :, :])
        image_features = torch.cat(
            [image_features1 - image_features2, image_features2 - image_features3], dim=-1
        )
        cat_embeddings = self.cat_encoder(batch)
        table_features = self.table_feature_linear(
            torch.cat([batch["numerical_features"].half(), cat_embeddings], dim=-1)
        )
        features = torch.cat([image_features, table_features], dim=-1)
        encoder_output = self.encoder(features)
        output = self.output_linear(torch.cat([encoder_output, table_features], dim=-1)).view(
            -1, self.seq_len, self.num_classes
        )
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)


class MotionCNN(nn.Module):
    def __init__(self, config: DictConfig):
        super(MotionCNN, self).__init__()
        self.image_model = timm.create_model(
            config.image_model, pretrained=True, in_chans=config.in_channels, num_classes=0
        )
        self.cat_encoder = CatEncoder()
        self.encoder = nn.Sequential(
            nn.Linear(self.image_model.num_features, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes
        self.output_linear = nn.Linear(config.hidden_size, self.num_classes * self.seq_len)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        image_features = self.image_model(batch["image"])
        encoder_output = self.encoder(image_features)
        output = self.output_linear(encoder_output).view(-1, self.seq_len, self.num_classes)
        return output

    def generate(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.forward(batch)
