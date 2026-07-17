from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def summarize_rollout(rollout, *, include_advantages: bool = False) -> dict[str, float]:
    """汇总 PPO/GRPO 共享的 rollout 指标。

    reward_mean 反映宽松奖励函数，accuracy 和 valid_option_rate 使用严格的
    <answer>...</answer> 解析。二者分开记录可以看出模型是否只答对了选项、
    但没有遵守输出格式。
    """
    response_lengths = rollout.response_mask.sum(dim=1).float()
    pred_letters = getattr(rollout, 'strict_pred_letters', rollout.pred_letters)
    denominator = max(len(pred_letters), 1)
    valid_option_rate = sum(letter is not None for letter in pred_letters) / denominator
    accuracy = sum(
        pred_letter == answer_key
        for pred_letter, answer_key in zip(pred_letters, rollout.answer_keys, strict=True)
    ) / denominator
    kl_mean = (
        (rollout.old_logprobs - rollout.ref_logprobs) * rollout.response_mask
    ).sum() / rollout.response_mask.sum().clamp_min(1)

    metrics = {
        'reward_mean': float(rollout.scores.mean().item()),
        'accuracy': float(accuracy),
        'valid_option_rate': float(valid_option_rate),
        'response_length_mean': float(response_lengths.mean().item()),
        'kl_mean': float(kl_mean.item()),
    }
    if include_advantages:
        metrics.update(
            advantage_mean=float(rollout.advantages.mean().item()),
            advantage_abs_mean=float(rollout.advantages.abs().mean().item()),
        )
    return metrics


@torch.no_grad()
def evaluate_rollouts(
    *,
    generate_rollout: Callable[..., Any],
    policy,
    reference_model,
    processor,
    valid_loader: Iterable[dict[str, Any]],
    generation_config,
    algorithm_config,
    algorithm_config_name: str,
    accelerator,
    max_batches: int | None = None,
) -> dict[str, float]:
    """执行 RL 公共评估循环；算法差异仅由 rollout 函数和配置参数注入。"""
    policy.eval()
    reward_sum = 0.0
    length_sum = 0.0
    valid = 0
    correct = 0
    total = 0

    for batch_index, batch in enumerate(valid_loader):
        rollout = generate_rollout(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            batch=batch,
            generation_config=generation_config,
            accelerator=accelerator,
            eval_mode=True,
            **{algorithm_config_name: algorithm_config},
        )
        pred_letters = getattr(rollout, 'strict_pred_letters', rollout.pred_letters)
        reward_sum += float(rollout.scores.sum().item())
        length_sum += float(rollout.response_mask.sum(dim=1).float().sum().item())
        valid += sum(letter is not None for letter in pred_letters)
        correct += sum(
            pred_letter == answer_key
            for pred_letter, answer_key in zip(
                pred_letters, rollout.answer_keys, strict=True
            )
        )
        total += len(pred_letters)
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    stats = torch.tensor(
        [reward_sum, length_sum, float(valid), float(correct), float(total)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    if accelerator.num_processes > 1:
        stats = accelerator.reduce(stats, reduction='sum')

    policy.train()
    total_count = max(float(stats[4].item()), 1.0)
    return {
        'reward_mean': float(stats[0].item() / total_count),
        'accuracy': float(stats[3].item() / total_count),
        'valid_option_rate': float(stats[2].item() / total_count),
        'response_length_mean': float(stats[1].item() / total_count),
    }

