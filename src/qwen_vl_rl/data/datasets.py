from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from torch.utils.data import Dataset

RecordT = TypeVar('RecordT')


@dataclass
class ThymeVLPPORecord:
    sample_id: int
    messages: list[dict[str, Any]]
    question: str
    choice_letter: str
    ground_truth: str
    reference_answer: str


@dataclass
class ThymeVLGRPORecord:
    sample_id: int
    prompt: list[dict[str, Any]]
    question: str
    choice_letter: str
    reward_target: str
    ground_truth: str


class ThymeVLPPOJsonlDataset(Dataset):
    def __init__(self, records: list[ThymeVLPPORecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            'sample_id': record.sample_id,
            'messages': copy.deepcopy(record.messages),
            'question': record.question,
            'choice_letter': record.choice_letter,
            'ground_truth': record.ground_truth,
            'reference_answer': record.reference_answer,
        }


class ThymeVLGRPOJsonlDataset(Dataset):
    def __init__(self, records: list[ThymeVLGRPORecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            'sample_id': record.sample_id,
            'prompt': copy.deepcopy(record.prompt),
            'question': record.question,
            'choice_letter': record.choice_letter,
            'reward_target': record.reward_target,
            'ground_truth': record.ground_truth,
        }


# 一个自定义的数据整理器（collate function），专门用于为 Qwen VL（视觉语言模型）准备 PPO 训练所需的批次数据。
# 主要作用是将原始样本列表（每个样本包含对话消息、图像信息等）整理成一个统一的字典，其中包含：
#       1. 模型可以直接使用的输入张量（通过 processor 处理）
#       2. 训练所需的元数据（如样本 ID、答案选项、原始问题）
def load_ppo_records(jsonl_path: str | Path) -> list[ThymeVLPPORecord]:
    records: list[ThymeVLPPORecord] = []
    path = Path(jsonl_path)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            payload = json.loads(line)
            choice_letter = (payload.get('choice_letter') or '').strip().upper()
            if choice_letter not in {'A', 'B', 'C', 'D'}:
                continue
            records.append(
                ThymeVLPPORecord(
                    sample_id=int(payload['id']),
                    messages=payload['messages'],
                    question=payload.get('question', ''),
                    choice_letter=choice_letter,
                    ground_truth=payload.get('ground_truth', ''),
                    reference_answer=payload.get('reference_answer', ''),
                )
            )
    return records


def load_grpo_records(jsonl_path: str | Path) -> list[ThymeVLGRPORecord]:
    records: list[ThymeVLGRPORecord] = []
    path = Path(jsonl_path)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            payload = json.loads(line)
            reward_target = (
                payload.get('reward_target') or payload.get('choice_letter') or ''
            ).strip().upper()
            if reward_target not in {'A', 'B', 'C', 'D'}:
                continue
            records.append(
                ThymeVLGRPORecord(
                    sample_id=int(payload['id']),
                    prompt=payload['prompt'],
                    question=payload.get('question', ''),
                    choice_letter=(payload.get('choice_letter') or reward_target).strip().upper(),
                    reward_target=reward_target,
                    ground_truth=payload.get('ground_truth', ''),
                )
            )
    return records


def _split_records(
    records: list[RecordT],
    *,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    source: str | Path,
    record_kind: str,
) -> tuple[list[RecordT], list[RecordT], list[RecordT]]:
    """以固定随机种子划分样本，确保三种训练路线使用相同的划分语义。"""
    total_needed = train_size + eval_size + test_size
    if len(records) < total_needed:
        raise ValueError(
            f'Not enough {record_kind} records: need {total_needed}, '
            f'found {len(records)} in {source}'
        )

    shuffled = list(records)
    random.Random(split_seed).shuffle(shuffled)
    train_records = shuffled[:train_size]
    eval_records = shuffled[train_size : train_size + eval_size]
    test_records = shuffled[train_size + eval_size : total_needed]

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_eval_samples is not None:
        eval_records = eval_records[:max_eval_samples]
    return train_records, eval_records, test_records


def create_split_datasets(
    jsonl_path: str | Path,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[ThymeVLPPOJsonlDataset, ThymeVLPPOJsonlDataset, ThymeVLPPOJsonlDataset]:
    records = load_ppo_records(jsonl_path)
    train_records, eval_records, test_records = _split_records(
        records,
        train_size=train_size,
        eval_size=eval_size,
        test_size=test_size,
        split_seed=split_seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        source=jsonl_path,
        record_kind='PPO',
    )

    return (
        ThymeVLPPOJsonlDataset(train_records),
        ThymeVLPPOJsonlDataset(eval_records),
        ThymeVLPPOJsonlDataset(test_records),
    )


def create_grpo_split_datasets(
    jsonl_path: str | Path,
    train_size: int,
    eval_size: int,
    test_size: int,
    split_seed: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[ThymeVLGRPOJsonlDataset, ThymeVLGRPOJsonlDataset, ThymeVLGRPOJsonlDataset]:
    records = load_grpo_records(jsonl_path)
    train_records, eval_records, test_records = _split_records(
        records,
        train_size=train_size,
        eval_size=eval_size,
        test_size=test_size,
        split_seed=split_seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        source=jsonl_path,
        record_kind='GRPO',
    )

    return (
        ThymeVLGRPOJsonlDataset(train_records),
        ThymeVLGRPOJsonlDataset(eval_records),
        ThymeVLGRPOJsonlDataset(test_records),
    )


