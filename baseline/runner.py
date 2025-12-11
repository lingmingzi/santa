import time
from typing import List, Optional

import numpy as np
from numba import cuda

from geometry import calc_side
from io_utils import load_csv, save_csv
from optimization import optimize_config


def optimize_pipeline(limit_n: Optional[List[int]] = None,
                      input_file: str = 'submission.csv',
                      output_file: str = 'submission.csv',
                      iters: int = 20,
                      restarts: int = 10):
    try:
        if not cuda.is_available():
            print("=" * 70)
            print("CUDA device NOT detected. Proceeding with CPU/Host functions.")
            print("Overlap checks will run via CPU fallback instead of GPU kernels.")
            print("=" * 70)
    except Exception as e:
        print(f"Error checking CUDA availability: {e}. Proceeding.")
    print(f"Loading {input_file}...", flush=True)
    configs = load_csv(input_file)
    all_tasks = sorted([n for n in configs])
    if limit_n:
        tasks = [n for n in all_tasks if n in limit_n]
        print(f"Loaded {len(configs)} configurations. Optimizing subset: {tasks}", flush=True)
    else:
        tasks = all_tasks
        print(f"Loaded {len(configs)} configurations. Optimizing all {len(tasks)} tasks.", flush=True)
    if not tasks:
        print("No tasks selected for optimization. Exiting.")
        return {}
    initial_score_all = sum(calc_side(configs[n]['x'], configs[n]['y'], configs[n]['deg'], n)**2 / n
                             for n in configs)
    print(f"Initial overall score (N=1 to 200): {initial_score_all:.6f}", flush=True)
    print(f"\nOptimization Parameters: SA Iters={iters}, Restarts={restarts}", flush=True)
    print("=" * 100, flush=True)
    print(f"| {'N':<3} | {'Initial Side':<14} | {'Optimized Side':<14} | {'Initial Score':<14} | {'Optimized Score':<14} | {'Improvement %':<15} | {'Running Total Score':<19} |", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    final_configs = configs.copy()
    current_total_score = initial_score_all
    for n in tasks:
        print(f"Starting N={n}...", flush=True)
        xs = configs[n]['x'][:n].copy()
        ys = configs[n]['y'][:n].copy()
        angs = configs[n]['deg'][:n].copy()
        old_side = calc_side(xs, ys, angs, n)
        old_score_contrib = old_side**2 / n
        current_restarts = restarts
        current_iters = iters
        if n <= 20:
            current_restarts = max(5, restarts // 2)
            current_iters = int(iters * 0.5)
        elif n > 150:
            current_restarts = restarts * 2
            current_iters = iters * 2
        opt_xs, opt_ys, opt_angs, new_side = optimize_config(n, xs, ys, angs, current_restarts, current_iters)
        new_score_contrib = new_side**2 / n
        current_total_score = current_total_score - old_score_contrib + new_score_contrib
        result_dict = {'x': np.zeros(200), 'y': np.zeros(200), 'deg': np.zeros(200)}
        result_dict['x'][:n] = opt_xs
        result_dict['y'][:n] = opt_ys
        result_dict['deg'][:n] = opt_angs
        final_configs[n] = result_dict
        improvement_pct = (old_score_contrib - new_score_contrib) / old_score_contrib * 100 if old_score_contrib > 1e-9 else 0.0
        status_msg = "Improved" if improvement_pct > 1e-4 else "No change"
        print(f"| {n:<3} | {old_side:<14.8f} | {new_side:<14.8f} | {old_score_contrib:<14.8f} | {new_score_contrib:<14.8f} | {improvement_pct:<15.4f} | {current_total_score:<19.8f} | {status_msg}", flush=True)
    elapsed = time.time() - t0
    print("=" * 100, flush=True)
    print(f"Total optimization time: {elapsed:.1f}s", flush=True)
    print(f"Final Score (All N):   {current_total_score:.6f}", flush=True)
    save_csv(output_file, final_configs)
    print(f"Saved optimized results to {output_file}", flush=True)
    return final_configs
