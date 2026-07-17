from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    DataConfig,
    GRPOTrainConfig,
    LoggingConfig,
    ModelConfig,
    PPOTrainConfig,
    SFTTrainConfig,
)


def _update_dataclass(instance: Any, updates: dict[str, Any], path: str = 'config') -> Any:
    """递归更新配置，并尽早拒绝拼写错误或放错层级的字段。"""
    known_fields = {item.name for item in fields(instance)}
    for key, value in updates.items():
        if key not in known_fields:
            raise ValueError(f'Unknown configuration field: {path}.{key}')
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value, path=f'{path}.{key}')
        else:
            setattr(instance, key, value)
    return instance


def _read_yaml(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'Configuration root must be a mapping: {path}')
    return payload


def _validate_common(config: Any) -> None:
    if not config.data.train_file:
        raise ValueError('config.data.train_file must not be empty')
    if not config.model.base_model_name_or_path:
        raise ValueError('config.model.base_model_name_or_path must not be empty')
    if config.num_train_epochs <= 0:
        raise ValueError('config.num_train_epochs must be greater than zero')

    for name in ('train_size', 'eval_size', 'test_size'):
        if getattr(config.data, name) < 0:
            raise ValueError(f'config.data.{name} must not be negative')
    if config.data.train_size <= 0:
        raise ValueError('config.data.train_size must be greater than zero')
    for name in ('max_train_samples', 'max_eval_samples', 'image_max_longest_edge'):
        value = getattr(config.data, name)
        if value is not None and value <= 0:
            raise ValueError(f'config.data.{name} must be greater than zero when set')
    for name in ('logging_steps', 'eval_steps', 'save_steps', 'save_total_limit'):
        if getattr(config.logging, name) <= 0:
            raise ValueError(f'config.logging.{name} must be greater than zero')
    if config.lora.r <= 0 or config.lora.alpha <= 0:
        raise ValueError('LoRA r and alpha must be greater than zero')
    if config.optimizer.learning_rate <= 0:
        raise ValueError('config.optimizer.learning_rate must be greater than zero')


def validate_config(config: SFTTrainConfig | PPOTrainConfig | GRPOTrainConfig) -> None:
    """校验训练前即可确定的配置约束，不触碰模型或数据文件。"""
    _validate_common(config)
    if isinstance(config, SFTTrainConfig):
        if config.sft.per_device_train_batch_size <= 0:
            raise ValueError('config.sft.per_device_train_batch_size must be greater than zero')
        if config.sft.per_device_eval_batch_size <= 0:
            raise ValueError('config.sft.per_device_eval_batch_size must be greater than zero')
        if config.sft.gradient_accumulation_steps <= 0:
            raise ValueError('config.sft.gradient_accumulation_steps must be greater than zero')
        if not 0.0 <= config.sft.warmup_ratio <= 1.0:
            raise ValueError('config.sft.warmup_ratio must be between zero and one')
    elif isinstance(config, PPOTrainConfig):
        if config.ppo.per_device_prompt_batch_size <= 0:
            raise ValueError('config.ppo.per_device_prompt_batch_size must be greater than zero')
        if config.ppo.per_device_minibatch_size <= 0 or config.ppo.ppo_epochs <= 0:
            raise ValueError('PPO minibatch size and epochs must be greater than zero')
        if config.generation.max_new_tokens <= 0 or config.generation.eval_max_new_tokens <= 0:
            raise ValueError('Generation token limits must be greater than zero')
        if not 0.0 <= config.ppo.cliprange < 1.0:
            raise ValueError('config.ppo.cliprange must be in [0, 1)')
    elif isinstance(config, GRPOTrainConfig):
        if config.grpo.per_device_prompt_batch_size <= 0:
            raise ValueError('config.grpo.per_device_prompt_batch_size must be greater than zero')
        if config.grpo.per_device_minibatch_size <= 0 or config.grpo.grpo_epochs <= 0:
            raise ValueError('GRPO minibatch size and epochs must be greater than zero')
        if config.grpo.num_generations < 2:
            raise ValueError('config.grpo.num_generations must be at least two')
        if config.generation.max_new_tokens <= 0 or config.generation.eval_max_new_tokens <= 0:
            raise ValueError('Generation token limits must be greater than zero')
        if not 0.0 <= config.grpo.cliprange < 1.0:
            raise ValueError('config.grpo.cliprange must be in [0, 1)')


def load_ppo_config(config_path: str | Path) -> PPOTrainConfig:
    config = PPOTrainConfig(
        data=DataConfig(train_file=''),
        model=ModelConfig(base_model_name_or_path=''),
    )
    _update_dataclass(config, _read_yaml(config_path))
    validate_config(config)
    return config


def load_grpo_config(config_path: str | Path) -> GRPOTrainConfig:
    config = GRPOTrainConfig(
        data=DataConfig(train_file=''),
        model=ModelConfig(base_model_name_or_path=''),
        logging=LoggingConfig(output_dir='outputs/grpo/default'),
    )
    _update_dataclass(config, _read_yaml(config_path))
    validate_config(config)
    return config


def load_sft_config(config_path: str | Path) -> SFTTrainConfig:
    config = SFTTrainConfig(
        data=DataConfig(train_file=''),
        model=ModelConfig(base_model_name_or_path=''),
    )
    _update_dataclass(config, _read_yaml(config_path))
    validate_config(config)
    return config



