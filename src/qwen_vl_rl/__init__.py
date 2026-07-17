"""QwenVL-RFT 的公共 Python API。

包初始化保持轻量。模型、PEFT 和数据集对象在真正访问时才导入，因此答案解析、
配置检查和绘图等轻量工具不再被迫加载完整训练依赖。
"""

from __future__ import annotations

from importlib import import_module

from .algorithms.reward import extract_choice_letter, score_choice_predictions
from .config import (
    GRPOTrainConfig,
    PPOTrainConfig,
    SFTTrainConfig,
    load_grpo_config,
    load_ppo_config,
    load_sft_config,
)

_LAZY_EXPORTS = {
    'QwenVLPPOCollator': ('.data', 'QwenVLPPOCollator'),
    'QwenVLGRPOCollator': ('.data', 'QwenVLGRPOCollator'),
    'ThymeVLPPOJsonlDataset': ('.data', 'ThymeVLPPOJsonlDataset'),
    'ThymeVLGRPOJsonlDataset': ('.data', 'ThymeVLGRPOJsonlDataset'),
    'create_split_datasets': ('.data', 'create_split_datasets'),
    'create_grpo_split_datasets': ('.data', 'create_grpo_split_datasets'),
    'PPOPolicyWithValueHead': ('.models.policy', 'PPOPolicyWithValueHead'),
    'build_policy_model': ('.models.policy', 'build_policy_model'),
    'build_reference_model': ('.models.policy', 'build_reference_model'),
    'build_lora_policy_backbone': ('.models.policy', 'build_lora_policy_backbone'),
    'save_lora_checkpoint': ('.models.policy', 'save_lora_checkpoint'),
    'save_policy_checkpoint': ('.models.policy', 'save_policy_checkpoint'),
    'QwenVLSFTCollator': ('.data.sft', 'QwenVLSFTCollator'),
    'ThymeVLSFTDataset': ('.data.sft', 'ThymeVLSFTDataset'),
    'create_sft_datasets_from_ppo_records': (
        '.data.sft',
        'create_sft_datasets_from_ppo_records',
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    'SFTTrainConfig',
    'PPOTrainConfig',
    'GRPOTrainConfig',
    'load_sft_config',
    'load_ppo_config',
    'load_grpo_config',
    'extract_choice_letter',
    'score_choice_predictions',
    *_LAZY_EXPORTS,
]
