import json
from types import SimpleNamespace

import torch

from qwen_vl_rl.training.rl import RLTrainingHooks, run_rl_training_loop


class _Accelerator:
    device = torch.device('cpu')
    num_processes = 1
    is_main_process = True

    @staticmethod
    def backward(loss):
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def test_rl_training_loop_owns_shared_lifecycle(tmp_path):
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    events: list[tuple[str, int | None]] = []

    hooks = RLTrainingHooks(
        generate_rollout=lambda batch: SimpleNamespace(
            sequences=torch.zeros((1, 1)),
            batch=batch,
        ),
        build_minibatch=lambda rollout, indices: {'indices': indices},
        compute_losses=lambda minibatch: {'loss': policy.weight.square().mean()},
        summarize_rollout=lambda rollout: {'reward_mean': float(rollout.batch['reward'])},
        evaluate=lambda: events.append(('eval', None)) or {'accuracy': 1.0},
        save_checkpoint=lambda step: events.append(('save', step)),
    )

    final_step = run_rl_training_loop(
        policy=policy,
        optimizer=optimizer,
        train_loader=[{'reward': 0.5}, {'reward': 1.0}, {'reward': -1.0}],
        accelerator=_Accelerator(),
        output_dir=tmp_path,
        hooks=hooks,
        algorithm_name='test',
        num_train_epochs=2,
        update_epochs=1,
        minibatch_size=1,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_steps=2,
        save_steps=2,
        total_steps=2,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    assert final_step == 2
    assert [record['phase'] for record in records] == ['train', 'train', 'eval']
    assert events == [('eval', None), ('save', 2)]
