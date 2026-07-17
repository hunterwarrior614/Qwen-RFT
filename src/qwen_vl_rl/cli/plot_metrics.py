#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vl_rl.reporting.plotting import render_metrics_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render a training curve image from metrics.jsonl.'
    )
    parser.add_argument(
        'metrics',
        type=str,
        help='Path to metrics.jsonl or a run output directory containing metrics.jsonl.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output image path. Defaults to <run_dir>/training_curve.png.',
    )
    parser.add_argument(
        '--kind',
        choices=['auto', 'sft', 'ppo', 'grpo'],
        default='auto',
        help='Metric schema to plot. Defaults to auto detection.',
    )
    parser.add_argument(
        '--rolling-window',
        type=int,
        default=1,
        help='Optional rolling mean window for train curves.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=160,
        help='Output image DPI.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = render_metrics_curve(
        metrics_path=args.metrics,
        output_path=args.output,
        kind=args.kind,
        rolling_window=max(1, args.rolling_window),
        dpi=args.dpi,
    )
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()

