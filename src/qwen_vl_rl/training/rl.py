from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .io import append_metric, log_metrics


@dataclass(frozen=True)
class RLTrainingHooks:
    """PPO/GRPO 注入公共训练循环的算法操作。"""

    generate_rollout: Callable[[dict[str, Any]], Any]
    build_minibatch: Callable[[Any, torch.Tensor], dict[str, torch.Tensor]]
    compute_losses: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    summarize_rollout: Callable[[Any], dict[str, float]]
    evaluate: Callable[[], dict[str, float]]
    save_checkpoint: Callable[[int], None]


def _mean_metrics_across_processes(
    accelerator,
    metrics: dict[str, float],
) -> dict[str, float]:
    """对各进程的训练指标取平均，避免多卡日志只反映 rank 0。"""
    if accelerator.num_processes <= 1 or not metrics:
        return metrics
    keys = list(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys],
        device=accelerator.device,
        dtype=torch.float64,
    )
    values = accelerator.reduce(values, reduction='mean')
    return {key: float(value) for key, value in zip(keys, values.tolist(), strict=True)}


def run_rl_training_loop(
    *,
    policy,
    optimizer,
    train_loader: Iterable[dict[str, Any]],
    accelerator,
    output_dir: Path,
    hooks: RLTrainingHooks,
    algorithm_name: str,
    num_train_epochs: int,
    update_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    total_steps: int,
    initial_step: int = 0,
) -> int:
    """运行 PPO/GRPO 共享生命周期，返回最终 global step。"""
    global_step = initial_step
    for epoch in range(num_train_epochs):
        if global_step >= total_steps:
            break
        policy.train()
        for batch in train_loader:
            if global_step >= total_steps:
                break

            # rollout 固定旧策略输出；随后可以对同一批样本做多轮更新。
            rollout = hooks.generate_rollout(batch)
            rollout_size = rollout.sequences.shape[0]
            update_metrics: dict[str, float] = {}
            for _ in range(update_epochs):
                permutation = torch.randperm(rollout_size, device=accelerator.device)
                for start in range(0, rollout_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    minibatch = hooks.build_minibatch(rollout, indices)
                    loss_dict = hooks.compute_losses(minibatch)

                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss_dict['loss'])
                    accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()
                    update_metrics = {
                        key: float(value.detach().float().item())
                        for key, value in loss_dict.items()
                    }

            global_step += 1
            train_metrics = hooks.summarize_rollout(rollout)
            train_metrics.update(update_metrics)
            train_metrics = _mean_metrics_across_processes(accelerator, train_metrics)
            train_metrics.update(
                epoch=float(epoch),
                step=float(global_step),
                total_steps=float(total_steps),
            )
            if accelerator.is_main_process:
                append_metric(output_dir, {'phase': 'train', **train_metrics})
            if global_step % logging_steps == 0 or global_step == 1:
                log_metrics(
                    accelerator,
                    prefix=f'{algorithm_name}/train',
                    metrics=train_metrics,
                    total_steps=total_steps,
                )

            if global_step % eval_steps == 0:
                eval_metrics = hooks.evaluate()
                eval_metrics.update(
                    step=float(global_step),
                    epoch=float(epoch),
                    total_steps=float(total_steps),
                )
                if accelerator.is_main_process:
                    append_metric(output_dir, {'phase': 'eval', **eval_metrics})
                log_metrics(
                    accelerator,
                    prefix=f'{algorithm_name}/eval',
                    metrics=eval_metrics,
                    total_steps=total_steps,
                )

            if global_step % save_steps == 0 and accelerator.is_main_process:
                hooks.save_checkpoint(global_step)

    return global_step

