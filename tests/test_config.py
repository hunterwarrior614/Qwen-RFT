from pathlib import Path

import pytest

from qwen_vl_rl.config import (
    GRPOTrainConfig,
    PPOTrainConfig,
    SFTTrainConfig,
    load_grpo_config,
    load_ppo_config,
    load_sft_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_training_configs_load_without_accessing_model_or_data():
    sft = load_sft_config(PROJECT_ROOT / 'configs/sft_qwen_vl_lora.yaml')
    ppo = load_ppo_config(PROJECT_ROOT / 'configs/ppo_qwen_vl_lora.yaml')
    grpo = load_grpo_config(PROJECT_ROOT / 'configs/grpo_qwen_vl_lora.yaml')

    assert isinstance(sft, SFTTrainConfig)
    assert isinstance(ppo, PPOTrainConfig)
    assert isinstance(grpo, GRPOTrainConfig)
    assert sft.sft.gradient_accumulation_steps == 2
    assert ppo.ppo.ppo_epochs == 2
    assert grpo.grpo.num_generations == 4


def test_config_loader_rejects_unknown_fields(tmp_path: Path):
    config_path = tmp_path / 'invalid.yaml'
    config_path.write_text(
        '''
data:
  train_file: train.jsonl
model:
  base_model_name_or_path: model
ppo:
  typo_clip_range: 0.2
''',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=r'config\.ppo\.typo_clip_range'):
        load_ppo_config(config_path)


def test_grpo_config_requires_multiple_generations(tmp_path: Path):
    config_path = tmp_path / 'invalid_grpo.yaml'
    config_path.write_text(
        '''
data:
  train_file: train.jsonl
model:
  base_model_name_or_path: model
grpo:
  num_generations: 1
''',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='at least two'):
        load_grpo_config(config_path)

