#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from qwen_vl_rl.algorithms.answering import extract_choice_letter
from qwen_vl_rl.config import load_sft_config
from qwen_vl_rl.data.sft import QwenVLSFTCollator, create_sft_datasets_from_ppo_records
from qwen_vl_rl.models.policy import build_lora_policy_backbone
from qwen_vl_rl.reporting.plotting import render_metrics_curve
from qwen_vl_rl.reporting.reports import write_test_results_from_loader
from qwen_vl_rl.training.io import (
    advance_scheduler_to_step,
    append_metric,
    estimate_total_training_steps,
    initialize_run_files,
    load_optimizer_state_if_available,
    load_rng_state_if_available,
    load_scheduler_state_if_available,
    prepare_checkpoint_dir,
    prune_old_checkpoints,
    resolve_resume_checkpoint,
    resume_step_from_checkpoint,
    save_optimizer_and_training_state,
    write_dataset_split,
)
from qwen_vl_rl.utils import (
    dump_json,
    ensure_dir,
    move_tensors_to_device,
    resolve_object_paths,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SFT warm start for Qwen2.5-VL LoRA on Thyme VQA data.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sft_qwen_vl_lora.yaml',
    )
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help=(
            'Resume from a checkpoint directory, its adapter/ subdirectory, or "latest" '
            'for the newest checkpoint under output_dir.'
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    policy,
    processor,
    valid_loader,
    accelerator,
    max_new_tokens: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    policy.eval()
    losses = []
    exact = 0
    total = 0
    for batch_index, batch in enumerate(valid_loader):
        model_inputs = move_tensors_to_device(batch['model_inputs'], accelerator.device)
        outputs = policy(**model_inputs)
        losses.append(float(outputs.loss.detach().float().item()))

        prompt_inputs = move_tensors_to_device(batch['prompt_inputs'], accelerator.device)
        generated = accelerator.unwrap_model(policy).generate(
            input_ids=prompt_inputs['input_ids'],
            attention_mask=prompt_inputs['attention_mask'],
            pixel_values=prompt_inputs['pixel_values'],
            image_grid_thw=prompt_inputs['image_grid_thw'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        for row_idx, target in enumerate(batch['target_texts']):
            prediction = processor.tokenizer.decode(
                generated[row_idx, prompt_length:], skip_special_tokens=True
            ).strip()
            pred_letter = extract_choice_letter(prediction, require_answer_tag=True)
            target_letter = extract_choice_letter(target, require_answer_tag=True)
            exact += int(pred_letter == target_letter)
            total += 1

        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    stats = torch.tensor(
        [sum(losses), float(len(losses)), float(exact), float(total)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    if accelerator.num_processes > 1:
        stats = accelerator.reduce(stats, reduction='sum')

    policy.train()
    return {
        'eval_loss': float(stats[0].item() / max(float(stats[1].item()), 1.0)),
        'eval_exact_match': float(stats[2].item() / max(float(stats[3].item()), 1.0)),
    }


def save_checkpoint(
    model,
    processor,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    save_total_limit: int,
) -> None:
    checkpoint_dir = prepare_checkpoint_dir(output_dir, step)
    model.save_pretrained(checkpoint_dir / 'adapter')
    processor.save_pretrained(checkpoint_dir / 'processor')
    save_optimizer_and_training_state(
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        training_state={'step': step},
        scheduler=scheduler,
    )
    prune_old_checkpoints(output_dir, save_total_limit)


def render_training_curve(output_dir: Path) -> None:
    render_metrics_curve(output_dir, kind='sft')


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else args.config
    config = load_sft_config(config_path)
    resolve_object_paths(config.data, PROJECT_ROOT, required_attrs=['train_file'])
    resolve_object_paths(
        config.model,
        PROJECT_ROOT,
        required_attrs=['base_model_name_or_path'],
        optional_attrs=['sft_adapter_path'],
    )
    resolve_object_paths(config.logging, PROJECT_ROOT, required_attrs=['output_dir'])
    set_seed(config.seed)

    output_dir = ensure_dir(config.logging.output_dir)
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
    )

    # 每经过 gradient_accumulation_steps 个 mini-batch 才执行一次参数更新，
    # 当显存不足以支持较大的批次时，通过梯度累积来模拟更大的有效批次。
    accelerator = Accelerator(
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        initialize_run_files(
            output_dir,
            config.to_dict(),
            resume_checkpoint=resume_checkpoint,
            reset_metrics=resume_checkpoint is None,
        )
    accelerator.wait_for_everyone()
    # 创建了一个多模态处理器（Processor），专门用于处理视觉语言模型（如 Qwen2.5-VL）的输入
    processor = AutoProcessor.from_pretrained(config.model.base_model_name_or_path)

    train_dataset, valid_dataset, test_dataset = create_sft_datasets_from_ppo_records(
        jsonl_path=config.data.train_file,
        train_size=config.data.train_size,
        eval_size=config.data.eval_size,
        test_size=config.data.test_size,
        split_seed=config.data.split_seed,
        max_train_samples=config.data.max_train_samples,
        max_eval_samples=config.data.max_eval_samples,
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

    # SFT、PPO、GRPO 共用同一个 Qwen-VL + LoRA 模型工厂。
    model_config = deepcopy(config.model)
    if resume_checkpoint is not None:
        model_config.sft_adapter_path = str(resume_checkpoint / 'adapter')
    model = build_lora_policy_backbone(model_config, config.lora)

    collator = QwenVLSFTCollator(
        processor,
        image_max_longest_edge=config.data.image_max_longest_edge,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.sft.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.sft.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.sft.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        eps=config.optimizer.adam_epsilon,
        weight_decay=config.optimizer.weight_decay,
    )

    total_steps = estimate_total_training_steps(
        num_batches=len(train_loader),
        num_train_epochs=config.num_train_epochs,
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    warmup_steps = max(1, int(total_steps * config.sft.warmup_ratio))
    # Accelerator.prepare 会将 scheduler 包装为 AcceleratedScheduler。
    # 在默认 split_batches=False 的多卡训练中，每次真实 optimizer step
    # 会推动底层 scheduler 前进 num_processes 次，因此这里需要按进程数放大
    # scheduler 看到的总步数和 warmup 步数。
    scheduler_total_steps = total_steps * accelerator.num_processes
    scheduler_warmup_steps = warmup_steps * accelerator.num_processes
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=scheduler_total_steps,
    )

    if accelerator.is_main_process:
        print(
            '[sft/setup] '
            f"train_samples={len(train_dataset)} "
            f"valid_samples={len(valid_dataset)} "
            f"per_device_train_batch_size={config.sft.per_device_train_batch_size} "
            f"grad_accum={config.sft.gradient_accumulation_steps} "
            f"total_steps={total_steps} "
            f"warmup_steps={warmup_steps} "
            f"scheduler_total_steps={scheduler_total_steps} "
            f"scheduler_warmup_steps={scheduler_warmup_steps} "
            f"num_processes={accelerator.num_processes} "
            f"process_index={accelerator.process_index}",
            flush=True,
        )

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader,
        scheduler,
    )

    global_step = 0
    if resume_checkpoint is not None:
        global_step = resume_step_from_checkpoint(resume_checkpoint)
        optimizer_loaded = load_optimizer_state_if_available(optimizer, resume_checkpoint)
        scheduler_loaded = load_scheduler_state_if_available(scheduler, resume_checkpoint)
        rng_loaded = load_rng_state_if_available(resume_checkpoint)
        scheduler_advanced_steps = 0
        if not scheduler_loaded:
            scheduler_advanced_steps = advance_scheduler_to_step(
                scheduler,
                global_step * accelerator.num_processes,
            )
        if accelerator.is_main_process:
            print(
                '[sft/resume] '
                f'checkpoint={resume_checkpoint} '
                f'step={global_step} '
                f'optimizer_loaded={optimizer_loaded} '
                f'scheduler_loaded={scheduler_loaded} '
                f'rng_loaded={rng_loaded} '
                f'scheduler_advanced_steps={scheduler_advanced_steps}',
                flush=True,
            )

    for epoch in range(config.num_train_epochs):
        if global_step >= total_steps:
            break
        model.train()
        for batch in train_loader:
            if global_step >= total_steps:
                break
            with accelerator.accumulate(model):
                model_inputs = move_tensors_to_device(batch['model_inputs'], accelerator.device)
                outputs = model(**model_inputs)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.optimizer.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    current_lr = float(scheduler.get_last_lr()[0])
                    train_record = {
                        'phase': 'train',
                        'step': global_step,
                        'epoch': epoch,
                        'loss': float(loss.detach().float().item()),
                        'lr': current_lr,
                        'total_steps': total_steps,
                    }
                    append_metric(output_dir, train_record)
                    if global_step % config.logging.logging_steps == 0 or global_step == 1 or global_step == total_steps:
                        print(
                            '[sft/train] '
                            f"step={global_step}/{total_steps} "
                            f"loss={train_record['loss']:.4f} "
                            f"lr={current_lr:.6e}",
                            flush=True,
                        )
                if global_step % config.logging.eval_steps == 0:
                    metrics = evaluate(
                        model,
                        processor,
                        valid_loader,
                        accelerator,
                        max_new_tokens=config.sft.max_new_tokens_eval,
                        max_batches=8,
                    )
                    if accelerator.is_main_process:
                        append_metric(
                            output_dir,
                            {
                                'phase': 'eval',
                                'step': global_step,
                                'epoch': epoch,
                                'eval_loss': float(metrics['eval_loss']),
                                'eval_exact_match': float(metrics['eval_exact_match']),
                                'total_steps': total_steps,
                            },
                        )
                        print(
                            '[sft/eval] '
                            f"step={global_step}/{total_steps} "
                            + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()),
                            flush=True,
                        )
                if (
                    global_step % config.logging.save_steps == 0
                    and accelerator.is_main_process
                ):
                    save_checkpoint(
                        accelerator.unwrap_model(model),
                        processor,
                        optimizer,
                        scheduler,
                        output_dir,
                        global_step,
                        config.logging.save_total_limit,
                    )
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    final_metrics = evaluate(
        model,
        processor,
        valid_loader,
        accelerator,
        max_new_tokens=config.sft.max_new_tokens_eval,
    )
    if accelerator.is_main_process:
        test_output = write_test_results_from_loader(
            policy=model,
            processor=processor,
            loader=test_loader,
            accelerator=accelerator,
            max_new_tokens=config.sft.max_new_tokens_eval,
            output_dir=output_dir,
        )
        append_metric(
            output_dir,
            {
                'phase': 'eval',
                'step': global_step,
                'epoch': config.num_train_epochs - 1,
                'eval_loss': float(final_metrics['eval_loss']),
                'eval_exact_match': float(final_metrics['eval_exact_match']),
                'total_steps': total_steps,
            },
        )
        print(
            '[sft/final_eval] '
            f"step={global_step}/{total_steps} "
            + ' '.join(f'{k}={v:.4f}' for k, v in final_metrics.items()),
            flush=True,
        )
        save_checkpoint(
            accelerator.unwrap_model(model),
            processor,
            optimizer,
            scheduler,
            output_dir,
            global_step,
            config.logging.save_total_limit,
        )
        dump_json(
            {
                'test_size': len(test_dataset),
                'global_step': global_step,
                'total_steps': total_steps,
                'final_eval': final_metrics,
                'test_metrics': test_output['metrics'],
                'test_results': test_output['paths'],
            },
            output_dir / 'train_summary.json',
        )
        render_training_curve(output_dir)


if __name__ == '__main__':
    main()
