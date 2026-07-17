#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from qwen_vl_rl.algorithms.ppo import build_minibatch, compute_ppo_losses, generate_rollout_batch
from qwen_vl_rl.config import load_ppo_config
from qwen_vl_rl.data import QwenVLPPOCollator, create_split_datasets
from qwen_vl_rl.models.policy import (
    build_policy_model,
    build_reference_model,
    save_policy_checkpoint,
)
from qwen_vl_rl.reporting.plotting import render_metrics_curve
from qwen_vl_rl.reporting.reports import write_test_results_from_loader
from qwen_vl_rl.training.io import (
    append_metric,
    estimate_total_training_steps,
    initialize_run_files,
    load_optimizer_state_if_available,
    load_rng_state_if_available,
    log_metrics,
    prepare_checkpoint_dir,
    prune_old_checkpoints,
    resolve_resume_checkpoint,
    resume_step_from_checkpoint,
    save_optimizer_and_training_state,
    write_dataset_split,
)
from qwen_vl_rl.training.metrics import evaluate_rollouts
from qwen_vl_rl.training.metrics import summarize_rollout as summarize_rl_rollout
from qwen_vl_rl.training.rl import RLTrainingHooks, run_rl_training_loop
from qwen_vl_rl.utils import dump_json, ensure_dir, resolve_object_paths, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL PPO with LoRA on Thyme VQA data.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ppo_qwen_vl_lora.yaml',
    )
    parser.add_argument('--max-steps', type=int, default=None, help='Optional cap on PPO prompt updates.')
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help=(
            'Resume training from a PPO checkpoint directory, its adapter/ subdirectory, '
            'or "latest" for the newest checkpoint under output_dir.'
        ),
    )
    parser.add_argument('--test-only', action='store_true', help='Only run test split and write test_results.')
    parser.add_argument(
        '--policy-adapter-path',
        type=str,
        default=None,
        help='Adapter path for evaluation. Accepts either an adapter dir or a checkpoint dir containing adapter/.',
    )
    return parser.parse_args()


def _normalize_adapter_path(path: Path) -> Path:
    if path.is_dir() and (path / 'adapter').is_dir():
        return path / 'adapter'
    return path


def resolve_test_policy_adapter_path(output_dir: Path, configured_sft_adapter_path: str | None, explicit_path: str | None) -> str | None:
    if explicit_path:
        return str(_normalize_adapter_path(Path(explicit_path).expanduser()).resolve())

    checkpoint_adapters: list[tuple[int, str]] = []
    for checkpoint_dir in output_dir.glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-(\d+)', checkpoint_dir.name)
        adapter_dir = checkpoint_dir / 'adapter'
        if match and adapter_dir.is_dir():
            checkpoint_adapters.append((int(match.group(1)), str(adapter_dir.resolve())))

    if checkpoint_adapters:
        checkpoint_adapters.sort(key=lambda item: item[0])
        return checkpoint_adapters[-1][1]

    if configured_sft_adapter_path:
        return str(_normalize_adapter_path(Path(configured_sft_adapter_path)).resolve())
    return None


def load_value_head_from_checkpoint(policy, checkpoint_dir: Path) -> None:
    value_head_path = checkpoint_dir / 'value_head.pt'
    if not value_head_path.exists():
        raise ValueError(
            f'PPO resume checkpoint is missing value_head.pt: {checkpoint_dir}'
        )
    policy.value_head.load_state_dict(
        torch.load(value_head_path, map_location='cpu', weights_only=True)
    )


def summarize_rollout(rollout) -> dict[str, float]:
    return summarize_rl_rollout(rollout)


@torch.no_grad()
def run_evaluation(
    policy,
    reference_model,
    processor,
    valid_loader,
    config,
    accelerator,
    max_batches: int | None = None,
) -> dict[str, float]:
    return evaluate_rollouts(
        generate_rollout=generate_rollout_batch,
        policy=policy,
        reference_model=reference_model,
        processor=processor,
        valid_loader=valid_loader,
        generation_config=config.generation,
        algorithm_config=config.ppo,
        algorithm_config_name='ppo_config',
        accelerator=accelerator,
        max_batches=max_batches,
    )


def render_training_curve(output_dir: Path) -> None:
    render_metrics_curve(output_dir, kind='ppo')


def save_checkpoint(policy, optimizer, output_dir: Path, step: int, config) -> None:
    unwrapped = policy
    if hasattr(policy, 'module'):
        unwrapped = policy.module
    checkpoint_dir = prepare_checkpoint_dir(output_dir, step)
    save_policy_checkpoint(
        policy=unwrapped,
        output_dir=checkpoint_dir,
        metadata={
            'step': step,
            'base_model': config.model.base_model_name_or_path,
        },
    )
    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={
            'step': step,
            'base_model': config.model.base_model_name_or_path,
            'output_dir': str(output_dir),
        },
    )
    prune_old_checkpoints(output_dir, config.logging.save_total_limit)
def main() -> None:
    args = parse_args()
    config = load_ppo_config(args.config)
    resolve_object_paths(
        config.data,
        PROJECT_ROOT,
        required_attrs=['train_file'],
    )
    resolve_object_paths(
        config.model,
        PROJECT_ROOT,
        required_attrs=['base_model_name_or_path'],
        optional_attrs=['sft_adapter_path'],
    )
    resolve_object_paths(
        config.logging,
        PROJECT_ROOT,
        required_attrs=['output_dir'],
    )
    set_seed(config.seed)

    output_dir = ensure_dir(config.logging.output_dir)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
    )
    if args.test_only:
        explicit_test_adapter = args.policy_adapter_path
        if resume_checkpoint is not None:
            explicit_test_adapter = str(resume_checkpoint)
        config.model.sft_adapter_path = resolve_test_policy_adapter_path(
            output_dir=output_dir,
            configured_sft_adapter_path=config.model.sft_adapter_path,
            explicit_path=explicit_test_adapter,
        )
        if config.model.sft_adapter_path is None:
            raise ValueError(
                'No adapter available for test-only evaluation. '
                'Pass --policy-adapter-path or train/save a PPO checkpoint first.'
            )

    # 创建了一个 Accelerator 对象
    # 其作用是自动处理分布式训练（多 GPU、TPU、混合精度等），同时简化设备管理、梯度累积和混合精度训练等代码
    accelerator = Accelerator(gradient_accumulation_steps=1)
    if accelerator.is_main_process:
        initialize_run_files(
            output_dir,
            config.to_dict(),
            resume_checkpoint=resume_checkpoint,
            reset_metrics=not args.test_only and resume_checkpoint is None,
        )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print('Loading processor and datasets...')

    # 创建了一个多模态处理器（Processor），专门用于处理视觉语言模型（如 Qwen2.5-VL）的输入
    processor = AutoProcessor.from_pretrained(config.model.base_model_name_or_path)

    train_dataset, valid_dataset, test_dataset = create_split_datasets(
        jsonl_path=config.data.train_file,
        train_size=config.data.train_size,
        eval_size=config.data.eval_size,
        test_size=config.data.test_size,
        split_seed=config.data.split_seed,
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples,
    )
    collator = QwenVLPPOCollator(
        processor,
        image_max_longest_edge=config.data.image_max_longest_edge,
    )
    if accelerator.is_main_process:
        write_dataset_split(
            output_dir,
            train_dataset,
            valid_dataset,
            test_dataset,
            config.data.split_seed,
        )
    accelerator.wait_for_everyone()

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.ppo.per_device_prompt_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if accelerator.is_main_process:
        print('Loading policy and reference models...')
        if args.test_only:
            print(f'[ppo/test-only] loading adapter from {config.model.sft_adapter_path}')

    policy_model_config = deepcopy(config.model)
    if resume_checkpoint is not None and not args.test_only:
        policy_model_config.sft_adapter_path = str(resume_checkpoint / 'adapter')

    # 获得策略网络 Actor
    policy = build_policy_model(policy_model_config, config.lora)
    if resume_checkpoint is not None and not args.test_only:
        load_value_head_from_checkpoint(policy, resume_checkpoint)

    if args.test_only:
        policy = accelerator.prepare(policy)
        if accelerator.is_main_process:
            test_output = write_test_results_from_loader(
                policy=policy,
                processor=processor,
                loader=test_loader,
                accelerator=accelerator,
                max_new_tokens=config.generation.eval_max_new_tokens,
                output_dir=output_dir,
            )
            print('Test metrics:', test_output['metrics'])
            dump_json(
                {
                    'test_size': len(test_dataset),
                    'test_metrics': test_output['metrics'],
                    'test_results': test_output['paths'],
                },
                output_dir / 'test_summary.json',
            )
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ppo.per_device_prompt_batch_size,
        shuffle=True,
        collate_fn=collator,  # 批处理函数，用于将单个样本列表合并成一个批次
        num_workers=config.data.num_workers,  # 用于数据加载的子进程数量
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.ppo.per_device_prompt_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Reference Model（参考模型） 是一个参数被冻结的、与策略模型结构相同的模型，
    # 主要用于计算 KL 散度惩罚项，防止策略模型在优化过程中过度偏离原始行为（例如语言生成风格或事实性）
    reference_model = build_reference_model(config.model)

    optimizer = AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_epsilon,
        weight_decay=config.optimizer.weight_decay,
    )

    total_steps = estimate_total_training_steps(
        num_batches=len(train_loader),
        num_train_epochs=config.num_train_epochs,
        num_processes=accelerator.num_processes,
        max_steps=args.max_steps,
    )
    policy, reference_model, optimizer, train_loader, valid_loader = accelerator.prepare(
        policy,
        reference_model,
        optimizer,
        train_loader,
        valid_loader,
    )
    reference_model.eval()
    global_step = 0
    if resume_checkpoint is not None:
        global_step = resume_step_from_checkpoint(resume_checkpoint)
        optimizer_loaded = load_optimizer_state_if_available(optimizer, resume_checkpoint)
        rng_loaded = load_rng_state_if_available(resume_checkpoint)
        if accelerator.is_main_process:
            print(
                '[ppo/resume] '
                f'checkpoint={resume_checkpoint} '
                f'step={global_step} '
                f'optimizer_loaded={optimizer_loaded} '
                f'rng_loaded={rng_loaded}',
                flush=True,
            )
    if accelerator.is_main_process:
        print(
            '[ppo/setup] '
            f'train_samples={len(train_dataset)} '
            f'valid_samples={len(valid_dataset)} '
            f'prompt_batch_size={config.ppo.per_device_prompt_batch_size} '
            f'minibatch_size={config.ppo.per_device_minibatch_size} '
            f'ppo_epochs={config.ppo.ppo_epochs} '
            f'total_steps={total_steps} '
            f'num_processes={accelerator.num_processes} '
            f'process_index={accelerator.process_index}',
            flush=True,
        )

    hooks = RLTrainingHooks(
        generate_rollout=lambda batch: generate_rollout_batch(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            batch=batch,
            generation_config=config.generation,
            ppo_config=config.ppo,
            accelerator=accelerator,
        ),
        build_minibatch=lambda rollout, indices: build_minibatch(
            rollout=rollout,
            indices=indices,
            whiten_advantages=config.ppo.whiten_advantages,
            device=accelerator.device,
        ),
        compute_losses=lambda minibatch: compute_ppo_losses(
            policy=policy,
            minibatch=minibatch,
            cliprange=config.ppo.cliprange,
            value_cliprange=config.ppo.value_cliprange,
            vf_coef=config.ppo.vf_coef,
            entropy_coef=config.ppo.entropy_coef,
        ),
        summarize_rollout=summarize_rollout,
        evaluate=lambda: run_evaluation(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            valid_loader=valid_loader,
            config=config,
            accelerator=accelerator,
        ),
        save_checkpoint=lambda step: save_checkpoint(
            policy, optimizer, output_dir, step, config
        ),
    )
    global_step = run_rl_training_loop(
        policy=policy,
        optimizer=optimizer,
        train_loader=train_loader,
        accelerator=accelerator,
        output_dir=output_dir,
        hooks=hooks,
        algorithm_name='ppo',
        num_train_epochs=config.num_train_epochs,
        update_epochs=config.ppo.ppo_epochs,
        minibatch_size=config.ppo.per_device_minibatch_size,
        max_grad_norm=config.optimizer.max_grad_norm,
        logging_steps=config.logging.logging_steps,
        eval_steps=config.logging.eval_steps,
        save_steps=config.logging.save_steps,
        total_steps=total_steps,
        initial_step=global_step,
    )

    accelerator.wait_for_everyone()
    final_eval = run_evaluation(
        policy=policy,
        reference_model=reference_model,
        processor=processor,
        valid_loader=valid_loader,
        config=config,
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        final_eval['step'] = float(global_step)
        final_eval['epoch'] = float(config.num_train_epochs - 1)
        final_eval['total_steps'] = float(total_steps)
        test_output = write_test_results_from_loader(
            policy=policy,
            processor=processor,
            loader=test_loader,
            accelerator=accelerator,
            max_new_tokens=config.generation.eval_max_new_tokens,
            output_dir=output_dir,
        )
        append_metric(output_dir, {'phase': 'eval', **final_eval})
        log_metrics(
            accelerator,
            prefix='final_eval',
            metrics=final_eval,
            total_steps=total_steps,
        )
        save_checkpoint(policy, optimizer, output_dir, global_step, config)
        dump_json(
            {
                'test_size': len(test_dataset),
                'global_step': global_step,
                'total_steps': total_steps,
                'final_eval': final_eval,
                'test_metrics': test_output['metrics'],
                'test_results': test_output['paths'],
            },
            output_dir / 'train_summary.json',
        )
        render_training_curve(output_dir)


if __name__ == '__main__':
    main()

