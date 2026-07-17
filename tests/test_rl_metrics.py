from types import SimpleNamespace

import pytest
import torch

from qwen_vl_rl.cli.train_grpo import (
    run_evaluation as run_grpo_evaluation,
)
from qwen_vl_rl.cli.train_grpo import (
    summarize_rollout as summarize_grpo_rollout,
)
from qwen_vl_rl.cli.train_ppo import (
    run_evaluation as run_ppo_evaluation,
)
from qwen_vl_rl.cli.train_ppo import (
    summarize_rollout as summarize_ppo_rollout,
)


def _build_rollout():
    return SimpleNamespace(
        response_mask=torch.ones((3, 2), dtype=torch.bool),
        pred_letters=['A', 'B', None],
        answer_keys=['A', 'C', 'D'],
        scores=torch.tensor([1.0, -0.25, -0.5]),
        old_logprobs=torch.zeros((3, 2)),
        ref_logprobs=torch.zeros((3, 2)),
        advantages=torch.tensor([0.5, -0.5, 0.0]),
    )


def test_ppo_summary_separates_reward_mean_from_accuracy():
    metrics = summarize_ppo_rollout(_build_rollout())

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_grpo_summary_separates_reward_mean_from_accuracy():
    metrics = summarize_grpo_rollout(_build_rollout())

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_summary_uses_strict_predictions_when_rollout_provides_them():
    rollout = _build_rollout()
    rollout.strict_pred_letters = ['A', None, None]

    metrics = summarize_ppo_rollout(rollout)

    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 1 / 3


def test_ppo_evaluation_uses_total_count_as_denominator(monkeypatch):
    rollout = _build_rollout()

    def fake_generate_rollout_batch(**kwargs):
        return rollout

    import qwen_vl_rl.cli.train_ppo as ppo_train

    monkeypatch.setattr(ppo_train, 'generate_rollout_batch', fake_generate_rollout_batch)

    metrics = run_ppo_evaluation(
        policy=_FakePolicy(),
        reference_model=None,
        processor=None,
        valid_loader=[{}],
        config=SimpleNamespace(generation=None, ppo=None),
        accelerator=_FakeAccelerator(),
    )

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


def test_grpo_evaluation_uses_total_count_as_denominator(monkeypatch):
    rollout = _build_rollout()

    def fake_generate_grpo_rollout_batch(**kwargs):
        return rollout

    import qwen_vl_rl.cli.train_grpo as grpo_train

    monkeypatch.setattr(grpo_train, 'generate_grpo_rollout_batch', fake_generate_grpo_rollout_batch)

    metrics = run_grpo_evaluation(
        policy=_FakePolicy(),
        reference_model=None,
        processor=None,
        valid_loader=[{}],
        config=SimpleNamespace(generation=None, grpo=None),
        accelerator=_FakeAccelerator(),
    )

    assert metrics['reward_mean'] == pytest.approx((1.0 - 0.25 - 0.5) / 3)
    assert metrics['accuracy'] == 1 / 3
    assert metrics['valid_option_rate'] == 2 / 3


class _FakePolicy:
    def eval(self):
        pass

    def train(self):
        pass


class _FakeAccelerator:
    device = torch.device('cpu')
    num_processes = 1
