from __future__ import annotations

import re
from collections.abc import Iterable

from .answering import (
    extract_answer_tag_content,
)
from .answering import (
    extract_choice_letter as _extract_choice_letter,
)

EXPLICIT_CHOICE_PATTERN = re.compile(
    r'\b(?:answer|option|choice)\s*(?:is|:|：)?\s*(?:probably\s+)?([A-Da-d])\b',
    re.IGNORECASE,
)
UPPERCASE_LETTER_TOKEN_PATTERN = re.compile(r'(?<![A-Za-z])([A-D])(?![A-Za-z])')

STRICT_CORRECT_REWARD = 1.0
VALID_WRONG_REWARD = -0.25
INVALID_REWARD = -0.5


def extract_choice_letter(text: str) -> str | None:
    return _extract_choice_letter(text, require_answer_tag=True)


def extract_relaxed_choice_letter(text: str) -> str | None:
    if not text:
        return None

    strict_tagged = extract_choice_letter(text)
    if strict_tagged:
        return strict_tagged

    tagged_content = extract_answer_tag_content(text)
    if tagged_content is not None:
        return _extract_choice_from_short_text(tagged_content)

    return _extract_choice_from_short_text(text)


def score_single_prediction(prediction: str, answer_key: str) -> tuple[float, str | None]:
    answer_key = (answer_key or '').strip().upper()
    pred_letter = extract_relaxed_choice_letter(prediction)
    if pred_letter is None:
        return INVALID_REWARD, None
    if pred_letter == answer_key:
        return STRICT_CORRECT_REWARD, pred_letter
    return VALID_WRONG_REWARD, pred_letter


def score_choice_predictions(predictions: Iterable[str], answer_keys: Iterable[str]) -> dict[str, list[float] | list[str | None]]:
    rewards: list[float] = []
    extracted: list[str | None] = []
    strict_extracted: list[str | None] = []
    for prediction, answer_key in zip(predictions, answer_keys, strict=True):
        reward, pred_letter = score_single_prediction(prediction, answer_key)
        extracted.append(pred_letter)
        strict_extracted.append(extract_choice_letter(prediction))
        rewards.append(reward)
    return {
        'rewards': rewards,
        'pred_letters': extracted,
        # Reward 可以使用宽松解析缓解稀疏奖励，但准确率必须遵守提示词规定的
        # <answer>...</answer> 格式，避免训练指标与最终测试报告口径不一致。
        'strict_pred_letters': strict_extracted,
    }


def _extract_choice_from_short_text(text: str) -> str | None:
    direct = _extract_choice_letter(text, require_answer_tag=False)
    if direct:
        return direct

    normalized = (text or '').strip()
    if len(normalized) > 80:
        return None

    explicit_match = EXPLICIT_CHOICE_PATTERN.search(normalized)
    if explicit_match:
        return explicit_match.group(1).upper()

    matches = [
        match.group(1).upper()
        for match in UPPERCASE_LETTER_TOKEN_PATTERN.finditer(normalized)
    ]
    unique_matches = set(matches)
    if len(unique_matches) == 1:
        return matches[0]
    return None

