#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from qwen_vl_rl.algorithms.grpo import (
    build_grpo_minibatch,
    compute_grpo_losses,
    generate_grpo_rollout_batch,
)
from qwen_vl_rl.config import load_grpo_config
from qwen_vl_rl.data import QwenVLGRPOCollator, create_grpo_split_datasets
from qwen_vl_rl.models.policy import (
    build_lora_policy_backbone,
    build_reference_model,
    save_lora_checkpoint,
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
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL GRPO with LoRA on Thyme VQA data.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/grpo_qwen_vl_lora.yaml',
    )
    parser.add_argument('--max-steps', type=int, default=None, help='Optional cap on GRPO prompt updates.')
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help=(
            'Resume training from a GRPO checkpoint directory, its adapter/ subdirectory, '
            'or "latest" for the newest checkpoint under output_dir.'
        ),
    )
    parser.add_argument('--test-only', action='store_true', help='Only run test split and write test_results.')
    return parser.parse_args()


def summarize_rollout(rollout) -> dict[str, float]:
    return summarize_rl_rollout(rollout, include_advantages=True)


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
        generate_rollout=generate_grpo_rollout_batch,
        policy=policy,
        reference_model=reference_model,
        processor=processor,
        valid_loader=valid_loader,
        generation_config=config.generation,
        algorithm_config=config.grpo,
        algorithm_config_name='grpo_config',
        accelerator=accelerator,
        max_batches=max_batches,
    )


def render_training_curve(output_dir: Path) -> None:
    render_metrics_curve(output_dir, kind='grpo')


def save_checkpoint(policy, optimizer, output_dir: Path, step: int, config) -> None:
    unwrapped = policy
    if hasattr(policy, 'module'):
        unwrapped = policy.module
    checkpoint_dir = prepare_checkpoint_dir(output_dir, step)
    save_lora_checkpoint(
        policy_model=unwrapped,
        output_dir=checkpoint_dir,
        metadata={
            'step': step,
            'base_model': config.model.base_model_name_or_path,
            'algorithm': 'grpo',
        },
    )
    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={
            'step': step,
            'base_model': config.model.base_model_name_or_path,
            'output_dir': str(output_dir),
            'algorithm': 'grpo',
        },
    )
    prune_old_checkpoints(output_dir, config.logging.save_total_limit)
def main() -> None:
    args = parse_args()
    config = load_grpo_config(args.config)
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
    if args.test_only and resume_checkpoint is not None:
        config.model.sft_adapter_path = str(resume_checkpoint / 'adapter')

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

    processor = AutoProcessor.from_pretrained(config.model.base_model_name_or_path)
    train_dataset, valid_dataset, test_dataset = create_grpo_split_datasets(
        jsonl_path=config.data.train_file,
        train_size=config.data.train_size,
        eval_size=config.data.eval_size,
        test_size=config.data.test_size,
        split_seed=config.data.split_seed,
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples,
    )
    collator = QwenVLGRPOCollator(
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
        batch_size=config.grpo.per_device_prompt_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if accelerator.is_main_process:
        print('Loading policy and reference models...')

    policy_model_config = deepcopy(config.model)
    if resume_checkpoint is not None and not args.test_only:
        policy_model_config.sft_adapter_path = str(resume_checkpoint / 'adapter')

    policy = build_lora_policy_backbone(policy_model_config, config.lora)

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
        batch_size=config.grpo.per_device_prompt_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.grpo.per_device_prompt_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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
                '[grpo/resume] '
                f'checkpoint={resume_checkpoint} '
                f'step={global_step} '
                f'optimizer_loaded={optimizer_loaded} '
                f'rng_loaded={rng_loaded}',
                flush=True,
            )
    if accelerator.is_main_process:
        print(
            '[grpo/setup] '
            f'train_samples={len(train_dataset)} '
            f'valid_samples={len(valid_dataset)} '
            f'prompt_batch_size={config.grpo.per_device_prompt_batch_size} '
            f'num_generations={config.grpo.num_generations} '
            f'minibatch_size={config.grpo.per_device_minibatch_size} '
            f'grpo_epochs={config.grpo.grpo_epochs} '
            f'total_steps={total_steps} '
            f'num_processes={accelerator.num_processes} '
            f'process_index={accelerator.process_index}',
            flush=True,
        )

    hooks = RLTrainingHooks(
        generate_rollout=lambda batch: generate_grpo_rollout_batch(
            policy=policy,
            reference_model=reference_model,
            processor=processor,
            batch=batch,
            generation_config=config.generation,
            grpo_config=config.grpo,
            accelerator=accelerator,
        ),
        build_minibatch=lambda rollout, indices: build_grpo_minibatch(
            rollout=rollout,
            indices=indices,
            device=accelerator.device,
        ),
        compute_losses=lambda minibatch: compute_grpo_losses(
            policy=policy,
            minibatch=minibatch,
            cliprange=config.grpo.cliprange,
            kl_coef=config.grpo.kl_coef,
            entropy_coef=config.grpo.entropy_coef,
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
        algorithm_name='grpo',
        num_train_epochs=config.num_train_epochs,
        update_epochs=config.grpo.grpo_epochs,
        minibatch_size=config.grpo.per_device_minibatch_size,
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

