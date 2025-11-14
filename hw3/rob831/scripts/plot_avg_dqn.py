import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalar_series_from_eventdir(event_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Read a scalar series (x=step, y=value) for a given tag from a TensorBoard event directory.

    Returns lists of steps and values sorted by step.
    """
    # If multiple event files exist, EventAccumulator can take the directory.
    accumulator = EventAccumulator(event_dir)
    accumulator.Reload()

    if tag not in accumulator.Tags().get('scalars', []):
        raise ValueError(f"Tag '{tag}' not found in event dir: {event_dir}")

    events = accumulator.Scalars(tag)
    steps = [e.step for e in events]
    values = [float(e.value) for e in events]

    # Ensure sorted by step (usually already sorted)
    order = np.argsort(steps)
    steps_sorted = [steps[i] for i in order]
    values_sorted = [values[i] for i in order]
    return steps_sorted, values_sorted


def get_xy_for_run(run_dir: str, preferred_x_tag: str, y_tag: str) -> Tuple[List[int], List[float]]:
    """Load a run and return (x, y) series with robust x-tag fallback.

    Fallback order for x-axis:
    1) preferred_x_tag if present
    2) 'Train_EnvstepsSoFar'
    3) 'TimeSinceStart'
    4) use steps from y_tag series
    """
    acc = EventAccumulator(run_dir)
    acc.Reload()

    available = set(acc.Tags().get('scalars', []))

    if y_tag not in available:
        raise ValueError(f"Tag '{y_tag}' not found in event dir: {run_dir}")

    y_events = acc.Scalars(y_tag)
    y_steps = [e.step for e in y_events]
    y_vals = [float(e.value) for e in y_events]

    x_candidate = None
    for tag in [preferred_x_tag, 'Train_EnvstepsSoFar', 'TimeSinceStart']:
        if tag in available:
            x_events = acc.Scalars(tag)
            x_candidate = [e.step for e in x_events]
            break

    if x_candidate is None:
        x_candidate = y_steps

    return x_candidate, y_vals


def collect_runs(base_data_dir: str, exp_prefix: str, env_name: str) -> List[str]:
    """Find run directories matching exp_prefix and env_name.

    Example dir name pattern created by run scripts:
    {exp_name}_{env_name}_{timestamp}
    where exp_name starts with exp_prefix (e.g., 'q1_dqn_' or 'q1_doubledqn_').
    """
    pattern = os.path.join(base_data_dir, f"{exp_prefix}*_{env_name}_*")
    run_dirs = sorted(glob.glob(pattern))
    return [d for d in run_dirs if os.path.isdir(d)]


def average_across_seeds(x_series: List[List[int]], y_series: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average across seeds without trimming to the shortest run.

    - First, for each run, trim to its own min(len(x), len(y)).
    - Then pad to the longest length across runs with NaNs and compute mean/std
      per index using only available values.

    Returns (x_mean, y_mean, y_std) as numpy arrays of length equal to the
    longest run after per-run trimming.
    """
    if len(x_series) == 0 or len(y_series) == 0:
        raise ValueError("No series provided to average.")

    trimmed_x, trimmed_y = [], []
    for xs, ys in zip(x_series, y_series):
        n = min(len(xs), len(ys))
        if n > 0:
            trimmed_x.append(np.asarray(xs[:n], dtype=np.float64))
            trimmed_y.append(np.asarray(ys[:n], dtype=np.float64))

    if len(trimmed_x) == 0:
        raise ValueError("All runs are empty after alignment.")

    max_len = max(arr.shape[0] for arr in trimmed_y)

    def pad_nan(arr: np.ndarray, target_len: int) -> np.ndarray:
        out = np.full((target_len,), np.nan, dtype=np.float64)
        out[:arr.shape[0]] = arr
        return out

    x_pad = np.stack([pad_nan(arr, max_len) for arr in trimmed_x], axis=0)
    y_pad = np.stack([pad_nan(arr, max_len) for arr in trimmed_y], axis=0)


    x_mean = np.nanmean(x_pad, axis=0)
    y_mean = np.nanmean(y_pad, axis=0)
    y_std = np.nanstd(y_pad, axis=0)

    # Drop trailing positions where all seeds are NaN (shouldn't happen due to max_len)
    # but keep any position with at least one value.
    valid_mask = ~np.isnan(y_mean)
    x_mean = x_mean[valid_mask]
    y_mean = y_mean[valid_mask]
    y_std = y_std[valid_mask]

    return x_mean, y_mean, y_std


def main():
    parser = argparse.ArgumentParser(description="Average DQN vs Double DQN returns across seeds and plot")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to hw3/data directory containing run folders')
    parser.add_argument('--env_name', type=str, default='LunarLander-v3', help='Environment name used in run directories')
    parser.add_argument('--dqn_prefix', type=str, default='q1_dqn_', help='Experiment name prefix for DQN runs')
    parser.add_argument('--ddqn_prefix', type=str, default='q1_doubledqn_', help='Experiment name prefix for Double DQN runs')
    parser.add_argument('--metric', type=str, default='Train_AverageReturn', choices=['Train_AverageReturn', 'Eval_AverageReturn'], help='Which scalar to average and plot')
    parser.add_argument('--x_tag', type=str, default='Train_EnvstepsSoFar', help='Scalar tag for x-axis (steps)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the figure (PNG). If not set, saves under data_dir.')
    args = parser.parse_args()

    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args.data_dir = os.path.realpath(os.path.join(script_dir, '../../data'))

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    dqn_runs = collect_runs(args.data_dir, args.dqn_prefix, args.env_name)
    ddqn_runs = collect_runs(args.data_dir, args.ddqn_prefix, args.env_name)

    if len(dqn_runs) == 0:
        raise RuntimeError(f"No DQN runs found matching prefix '{args.dqn_prefix}' and env '{args.env_name}' in {args.data_dir}")
    if len(ddqn_runs) == 0:
        raise RuntimeError(f"No Double DQN runs found matching prefix '{args.ddqn_prefix}' and env '{args.env_name}' in {args.data_dir}")

    # Read scalar series from each run (with x-tag fallback)
    dqn_xs, dqn_ys = [], []
    for run_dir in dqn_runs:
        x, y = get_xy_for_run(run_dir, args.x_tag, args.metric)
        dqn_xs.append(x)
        dqn_ys.append(y)

    ddqn_xs, ddqn_ys = [], []
    for run_dir in ddqn_runs:
        x, y = get_xy_for_run(run_dir, args.x_tag, args.metric)
        ddqn_xs.append(x)
        ddqn_ys.append(y)

    # Average across seeds (align by index up to min length)
    dqn_x_mean, dqn_y_mean, dqn_y_std = average_across_seeds(dqn_xs, dqn_ys)
    ddqn_x_mean, ddqn_y_mean, ddqn_y_std = average_across_seeds(ddqn_xs, ddqn_ys)


    plt.figure(figsize=(8, 5))
    plt.plot(dqn_x_mean, dqn_y_mean, label='DQN', color='#1f77b4')
    plt.fill_between(dqn_x_mean, dqn_y_mean - dqn_y_std, dqn_y_mean + dqn_y_std, color='#1f77b4', alpha=0.2)

    plt.plot(ddqn_x_mean, ddqn_y_mean, label='Double DQN', color='#ff7f0e')
    plt.fill_between(ddqn_x_mean, ddqn_y_mean - ddqn_y_std, ddqn_y_mean + ddqn_y_std, color='#ff7f0e', alpha=0.2)

    plt.xlabel('Env Steps')
    plt.ylabel(args.metric.replace('_', ' '))
    plt.title(f"{args.env_name}: DQN vs Double DQN (avg over {min(len(dqn_runs), len(ddqn_runs))} seeds)")
    plt.legend()
    plt.tight_layout()

    if args.save_path is None:
        args.save_path = os.path.join(args.data_dir, f"{args.env_name}_dqn_vs_doubledqn_avg.png")
    plt.savefig(args.save_path, dpi=150)
    print(f"Saved figure to: {args.save_path}")


if __name__ == '__main__':
    main()
