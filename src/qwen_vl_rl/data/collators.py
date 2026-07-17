from __future__ import annotations

import copy
from typing import Any

from ..utils import decode_first_image_from_messages


def prepare_tokenizer_for_padding(processor, padding_side: str) -> None:
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.pad_token
    )


def build_generation_prompt_texts(processor, batch: list[dict[str, Any]]) -> list[str]:
    return [
        processor.apply_chat_template(
            sample['messages'],
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in batch
    ]


def decode_prompt_images(
    batch: list[dict[str, Any]],
    image_max_longest_edge: int | None = None,
) -> list[Any]:
    return [
        decode_first_image_from_messages(
            sample['messages'],
            image_max_longest_edge=image_max_longest_edge,
        )
        for sample in batch
    ]


def build_processor_inputs(processor, texts: list[str], images: list[Any]):
    return build_processor_inputs_with_padding_side(
        processor,
        texts=texts,
        images=images,
        padding_side=processor.tokenizer.padding_side,
    )


def build_processor_inputs_with_padding_side(
    processor,
    texts: list[str],
    images: list[Any],
    padding_side: str,
):
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = padding_side
    try:
        return processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors='pt',
        )
    finally:
        processor.tokenizer.padding_side = original_padding_side


def collect_prompt_metadata(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        'sample_ids': [sample['sample_id'] for sample in batch],
        'messages': [sample['messages'] for sample in batch],
        'questions': [sample['question'] for sample in batch],
    }




class QwenVLPPOCollator:
    def __init__(self, processor, image_max_longest_edge: int | None = None):
        self.processor = processor
        self.image_max_longest_edge = image_max_longest_edge

        # 生成任务中，需要将不同长度的 prompt 在左侧补齐，这样可以保证生成时新 token 追加在右侧，且 attention mask 计算正确
        prepare_tokenizer_for_padding(self.processor, padding_side='left')

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = build_generation_prompt_texts(self.processor, batch)
        prompt_images = decode_prompt_images(
            batch,
            image_max_longest_edge=self.image_max_longest_edge,
        )
        metadata = collect_prompt_metadata(batch)
        answer_keys = []
        ground_truths = []

        for sample in batch:
            answer_keys.append(sample['choice_letter'])
            ground_truths.append(sample.get('ground_truth', sample['choice_letter']))

        inputs = build_processor_inputs(self.processor, prompt_texts, prompt_images)
        return {
            'sample_ids': metadata['sample_ids'],
            'answer_keys': answer_keys,
            'questions': metadata['questions'],
            'ground_truths': ground_truths,
            'messages': [copy.deepcopy(messages) for messages in metadata['messages']],
            'prompt_texts': prompt_texts,
            'prompt_images': prompt_images,
            'prompt_inputs': inputs,
        }
    """
    假设 batch size 为 2, 则经过 __call__ 整理的 batch 内容形如：
    {
        "sample_ids": [1001, 1002],
        "answer_keys": ["B", "A"],
        "questions": ["What is the color of the car?", "How many apples are there?"],
        "prompt_texts": [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision|>What is the color of the car?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision|>How many apples are there?<|im_end|>\n<|im_start|>assistant\n"
        ],
        "prompt_images": [<PIL.Image>, <PIL.Image>],
        "prompt_inputs": {
            "input_ids": torch.tensor([[151644, 151645, ..., 151649], [151644, 151645, ..., 151649]]),
            "attention_mask": torch.tensor([[1, 1, ..., 0], [1, 1, ..., 0]]),
            "pixel_values": torch.tensor([...]),  // shape: (total_patches, 3, 448, 448)
            "image_grid_thw": torch.tensor([[1, 28, 28], [1, 28, 28]])
        }
    }
    """


class QwenVLGRPOCollator(QwenVLPPOCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ppo_batch = [
            {
                'sample_id': sample['sample_id'],
                'messages': sample['prompt'],
                'question': sample['question'],
                'choice_letter': sample['reward_target'],
                'ground_truth': sample['ground_truth'],
            }
            for sample in batch
        ]
        output = super().__call__(ppo_batch)
        output['reward_targets'] = [sample['reward_target'] for sample in batch]
        return output


