import dataclasses
from typing import Dict, Any

@dataclasses.dataclass
class EncoderConfig:
    kernel_size: int = 16
    out_channels: int = 256

@dataclasses.dataclass
class BackboneConfig:
    dim: int = 256
    num_blocks: int = 16
    num_heads: int = 4
    ffn_dim: int = 1024
    fsmn_kernel_size: int = 3
    num_groups: int = 32

@dataclasses.dataclass
class CountingHeadConfig:
    qmax: int = 5
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 1024

@dataclasses.dataclass
class MaskingHeadConfig:
    qmax: int = 5
    out_dim: int = 256

@dataclasses.dataclass
class LossConfig:
    lambda_count: float = 1.0

@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 2
    lr: float = 4e-4
    max_epochs: int = 200
    sample_rate: int = 8000
    clip_grad_norm: float = 5.0

@dataclasses.dataclass
class MTCNetConfig:
    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    backbone: BackboneConfig = dataclasses.field(default_factory=BackboneConfig)
    counting_head: CountingHeadConfig = dataclasses.field(default_factory=CountingHeadConfig)
    masking_head: MaskingHeadConfig = dataclasses.field(default_factory=MaskingHeadConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MTCNetConfig":
        def _build_dataclass(target_class, source_dict):
            if not isinstance(source_dict, dict):
                return source_dict
            fields = {f.name: f.type for f in dataclasses.fields(target_class)}
            kwargs = {}
            for name, val in source_dict.items():
                if name in fields:
                    field_type = fields[name]
                    if dataclasses.is_dataclass(field_type):
                        kwargs[name] = _build_dataclass(field_type, val)
                    else:
                        kwargs[name] = val
            return target_class(**kwargs)

        return _build_dataclass(cls, d)
