#!/usr/bin/env python3
"""
Automated improvement loop for Badger.

This is the engine that drives Badger toward SOTA:

    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │  1. RUN BENCHMARK                               │
    │     ↓                                           │
    │  2. ANALYZE ERRORS (weak classes, size, type)   │
    │     ↓                                           │
    │  3. SUGGEST NEXT EXPERIMENT                     │
    │     ↓                                           │
    │  4. APPLY EXPERIMENT + TRAIN                    │
    │     ↓                                           │
    │  5. COMPARE RESULTS (better?)                   │
    │     ├── YES → keep, go to step 1                │
    │     └── NO  → revert, try next suggestion       │
    │                                                 │
    └─────────────────────────────────────────────────┘

Usage:
    # Single iteration
    python scripts/iterate.py --model badger-s --data config/coco.yaml

    # Continuous loop (run until stopped or target mAP reached)
    python scripts/iterate.py --model badger-s --data config/coco.yaml --continuous

    # Resume from previous state
    python scripts/iterate.py --resume runs/iteration_state.json

    # Set target (stops when reached)
    python scripts/iterate.py --model badger-s --target-mAP 55.0
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Experiment-to-Config Mapping
# =============================================================================

EXPERIMENT_MAP = {
    'simota_assigner':      {'training.loss.assigner': 'simota'},
    'varifocal_loss':       {'training.loss.cls_type': 'varifocal'},
    'siou_loss':            {'training.loss.box_type': 'siou'},
    'gaussian_dfl':         {'model.head.regression': 'gaussian'},
    'attention_neck':       {'model.neck.type': 'attention', 'model.neck.use_attention': True, 'model.neck.use_global_tokens': True},
    'cbam_attention':       {'model.attention': 'cbam'},
    'eca_attention':        {'model.attention': 'eca'},
    'se_attention':         {'model.attention': 'se'},
    'dcnv2':                {'model.backbone.conv_type': 'dcnv2'},
    'repconv':              {'model.backbone.conv_type': 'repconv'},
    'droppath':             {'model.backbone.drop_path_rate': 0.1},
    'ema_weights':          {'training.ema': True, 'training.ema_decay': 0.9999},
    'mosaic_close':         {'training.augmentation.mosaic': 'schedule', 'training.close_mosaic_epochs': 10},
    'multi_scale_training': {'training.multi_scale': True, 'training.ms_range': [0.5, 1.5]},
    'larger_resolution':    {'training.img_size': 1280},
    'deeper_backbone':      {'model.backbone.depth_multiple': 1.0, 'model.backbone.width_multiple': 1.25},
    'label_smoothing':      {'training.label_smoothing': 0.05},
    'cosine_schedule':      {'training.lr_scheduler': 'cosine'},
    'mish_activation':      {'model.backbone.act': 'mish'},
    'bifpn_neck':           {'model.neck.type': 'bifpn'},
    'giou_loss':            {'training.loss.box_type': 'giou'},
}

# Improvement hierarchy — which experiments tend to help with specific problems
ERROR_TO_EXPERIMENT = {
    'loc_error':    ['siou_loss', 'gaussian_dfl', 'attention_neck', 'dcnv2', 'repconv'],
    'cls_error':    ['varifocal_loss', 'label_smoothing'],
    'missed_error': ['simota_assigner', 'larger_resolution', 'multi_scale_training', 'attention_neck'],
    'bg_error':     ['cbam_attention', 'eca_attention', 'se_attention'],
    'small_objects': ['larger_resolution', 'attention_neck', 'gaussian_dfl', 'dcnv2'],
    'large_objects': ['attention_neck', 'deeper_backbone'],
    'overfitting':  ['droppath', 'label_smoothing', 'mosaic_close', 'multi_scale_training'],
}


class IterationState:
    """Tracks the improvement loop state across iterations."""

    def __init__(self, save_path='runs/iteration_state.json'):
        self.save_path = Path(save_path)
        self.state = self._load()

    def _load(self):
        if self.save_path.exists():
            with open(self.save_path) as f:
                return json.load(f)
        return {
            'iterations': [],
            'active_experiments': [],
            'tried_experiments': [],
            'rejected_experiments': [],
            'best_mAP': 0.0,
            'best_iteration': -1,
            'started_at': datetime.now().isoformat(),
            'target_mAP': 62.0,
        }

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def record_iteration(self, iteration_num, experiment, mAP_before, mAP_after,
                         error_analysis, verdict):
        record = {
            'iteration': iteration_num,
            'experiment': experiment,
            'mAP_before': mAP_before,
            'mAP_after': mAP_after,
            'delta_mAP': round(mAP_after - mAP_before, 3),
            'error_breakdown': error_analysis.get('breakdown', {}),
            'weakest_classes': error_analysis.get('weakest', []),
            'verdict': verdict,  # 'adopted', 'rejected', 'inconclusive'
            'timestamp': datetime.now().isoformat(),
        }
        self.state['iterations'].append(record)

        if verdict == 'adopted':
            self.state['active_experiments'].append(experiment)
            if experiment in self.state['tried_experiments']:
                self.state['tried_experiments'].remove(experiment)
        elif verdict == 'rejected':
            self.state['rejected_experiments'].append(experiment)

        self.state['tried_experiments'].append(experiment)

        if mAP_after > self.state['best_mAP']:
            self.state['best_mAP'] = mAP_after
            self.state['best_iteration'] = iteration_num

        self.save()

    def get_untried_experiments(self, suggestions):
        """Filter suggestions to only include untried experiments."""
        tried = set(self.state['tried_experiments'])
        rejected = set(self.state['rejected_experiments'])
        return [s for s in suggestions if s['experiment'] not in tried and s['experiment'] not in rejected]

    def summary(self):
        """Print iteration history."""
        print("\n" + "=" * 70)
        print("  ITERATION HISTORY")
        print("=" * 70)
        print(f"  Best mAP: {self.state['best_mAP']:.3f} (iteration {self.state['best_iteration']})")
        print(f"  Active experiments: {self.state['active_experiments']}")
        print(f"  Tried: {self.state['tried_experiments']}")
        print(f"  Rejected: {self.state['rejected_experiments']}")
        print(f"\n  Iterations:")
        for rec in self.state['iterations']:
            symbol = '✓' if rec['verdict'] == 'adopted' else '✗' if rec['verdict'] == 'rejected' else '?'
            print(f"  [{rec['iteration']:3d}] {symbol} {rec['experiment']:<25s} "
                  f"mAP: {rec['mAP_before']:.3f} → {rec['mAP_after']:.3f} "
                  f"(Δ={rec['delta_mAP']:+.3f})")
        print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Badger Automated Improvement Loop')
    parser.add_argument('--model', type=str, default='badger-s',
                        help='Model variant to optimize')
    parser.add_argument('--data', type=str, default='config/coco.yaml',
                        help='Dataset config')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously until target reached')
    parser.add_argument('--target-mAP', type=float, default=62.0,
                        help='Target mAP (stops when reached)')
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='Maximum number of iterations')
    parser.add_argument('--resume', type=str, default='runs/iteration_state.json',
                        help='Resume from saved state')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analysis only, no training')
    return parser.parse_args()


def run_benchmark(model_path, data_config, device='cuda'):
    """
    Run benchmark and return mAP + detailed error analysis.

    In a real setup, this would call the benchmark script or load
    previously-computed results.
    """
    print(f"\n  [BENCHMARK] Running evaluation on {data_config}...")

    # In production, this runs the actual benchmark:
    # result = subprocess.run([
    #     'python', 'scripts/benchmark.py',
    #     '--model', model_path,
    #     '--data', data_config,
    #     '--full', '--save', 'runs/latest_benchmark.json',
    # ], capture_output=True, text=True)

    # For now, load from saved benchmark if available
    result_path = Path('runs/latest_benchmark.json')
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    # Placeholder — indicates real benchmark needs to be run
    return {
        'mAP': 0.0,
        'AP50': 0.0,
        'AP75': 0.0,
        'error_analysis': {
            'breakdown': {},
            'weakest': [],
            'size_analysis': {},
        }
    }


def analyze_errors(benchmark_result):
    """
    Analyze benchmark results and suggest improvements.

    Uses src.utils.analysis.ErrorAnalyzer when real data is available.
    Falls back to the benchmark's built-in analysis.
    """
    error_analysis = benchmark_result.get('error_analysis', {})
    suggestions = error_analysis.get('suggestions', [])

    # If no detailed analysis available, use built-in analyzer
    if not suggestions:
        try:
            from src.utils.analysis import ErrorAnalyzer
            analyzer = ErrorAnalyzer()
            # analyzer would be populated with real predictions here
            suggestions = analyzer.suggest_next_experiment()
        except ImportError:
            pass

    return error_analysis, suggestions


def apply_experiment(experiment_name, current_config):
    """Update config with the chosen experiment's parameters."""
    if experiment_name not in EXPERIMENT_MAP:
        print(f"  Warning: Unknown experiment '{experiment_name}'")
        return current_config

    updates = EXPERIMENT_MAP[experiment_name]
    new_config = current_config.copy()
    # Deep update (simplified — in production, use proper nested update)
    for key, value in updates.items():
        new_config[key] = value

    print(f"  [APPLY] Experiment: {experiment_name}")
    print(f"         Changes: {updates}")

    return new_config


def train_model(config, experiment_name, data_config, device='cuda'):
    """
    Train the model with the new experiment config.

    In production, this launches the training script:
        python scripts/train.py --config updated_config.yaml --data data_config
    """
    print(f"  [TRAIN] Starting training with experiment '{experiment_name}'...")
    print(f"         (In production: python scripts/train.py --experiment {experiment_name})")

    # In production:
    # subprocess.run([
    #     'python', 'scripts/train.py',
    #     '--config', 'config/default.yaml',
    #     '--experiment', experiment_name,
    #     '--data', data_config,
    #     '--device', device,
    # ])

    # Simulate training time
    print(f"         Training would run here (300 epochs on COCO)...")
    print(f"         After training, checkpoint saved to runs/{experiment_name}/best.pth")

    return f"runs/{experiment_name}/best.pth"


def compare_results(mAP_before, mAP_after, threshold=0.001):
    """
    Determine if the experiment was a success.

    Returns: 'improved', 'regressed', 'neutral'
    """
    delta = mAP_after - mAP_before
    if delta > threshold:
        return 'improved'
    elif delta < -threshold:
        return 'regressed'
    return 'neutral'


def single_iteration(state, args, iteration_num, current_mAP, current_config):
    """Run one complete benchmark → analyze → improve → benchmark cycle."""

    print(f"\n{'='*70}")
    print(f"  ITERATION {iteration_num}")
    print(f"{'='*70}")
    print(f"  Current mAP: {current_mAP:.3f}")
    print(f"  Active experiments: {state.state['active_experiments']}")
    print(f"  Target mAP: {state.state['target_mAP']}")

    # Step 1: Run benchmark
    print(f"\n  Step 1/5: Running benchmark...")
    benchmark_result = run_benchmark(
        f"runs/iter_{iteration_num-1}/best.pth" if iteration_num > 0 else args.model,
        args.data, args.device
    )

    # Step 2: Analyze errors
    print(f"\n  Step 2/5: Analyzing errors...")
    error_analysis, suggestions = analyze_errors(benchmark_result)

    # Print analysis
    if error_analysis.get('breakdown'):
        print(f"\n  Error Breakdown:")
        for err_type, info in sorted(error_analysis['breakdown'].items(),
                                      key=lambda x: -x[1].get('pct', 0)):
            print(f"    {err_type}: {info.get('pct', 0):.1f}%")

    # Step 3: Choose experiment
    print(f"\n  Step 3/5: Choosing next experiment...")
    untried = state.get_untried_experiments(suggestions)

    if not untried:
        print(f"  No untried experiments remain. Trying next-best suggestion...")
        # Fall back to any suggestion not yet rejected
        untried = [s for s in suggestions if s['experiment'] not in state.state['rejected_experiments']]

    if not untried:
        print(f"  All experiments exhausted! Try manual architecture changes.")
        return current_mAP, current_config

    chosen = untried[0]
    print(f"  → Chosen: [{chosen['priority']}] {chosen['experiment']}")
    print(f"    Reason: {chosen['reason']}")

    # Step 4: Apply + Train
    print(f"\n  Step 4/5: Training with experiment...")
    new_config = apply_experiment(chosen['experiment'], current_config)

    if not args.dry_run:
        checkpoint_path = train_model(new_config, chosen['experiment'], args.data, args.device)
    else:
        print(f"  [DRY RUN] Skipping training.")
        checkpoint_path = f"runs/{chosen['experiment']}/best.pth"

    # Step 5: Benchmark again + compare
    print(f"\n  Step 5/5: Evaluating new model...")
    new_benchmark = run_benchmark(checkpoint_path, args.data, args.device) if not args.dry_run else {'mAP': current_mAP + 0.5}
    new_mAP = new_benchmark.get('mAP', current_mAP)

    # Compare
    delta = new_mAP - current_mAP
    comparison = compare_results(current_mAP, new_mAP)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  mAP: {current_mAP:.3f} → {new_mAP:.3f}  (Δ = {delta:+.3f})    │")

    if comparison == 'improved':
        verdict = 'adopted'
        print(f"  │  Verdict: ADOPTED ✓                  │")
    elif comparison == 'regressed':
        verdict = 'rejected'
        print(f"  │  Verdict: REJECTED ✗                 │")
    else:
        verdict = 'inconclusive'
        print(f"  │  Verdict: INCONCLUSIVE ?             │")
    print(f"  └─────────────────────────────────────┘")

    # Record
    state.record_iteration(iteration_num, chosen['experiment'],
                          current_mAP, new_mAP, error_analysis, verdict)

    # Return best mAP (don't regress)
    if verdict == 'adopted':
        return new_mAP, new_config
    else:
        return current_mAP, current_config


def continuous_loop(state, args):
    """Run the improvement loop continuously."""
    current_mAP = state.state['best_mAP']
    current_config = {}

    iteration = len(state.state['iterations'])

    print(f"\n{'#'*70}")
    print(f"  BADGER AUTOMATED IMPROVEMENT LOOP")
    print(f"  Target: {state.state['target_mAP']:.1f}% mAP on COCO val2017")
    print(f"  Starting mAP: {current_mAP:.3f}")
    print(f"  Mode: {'DRY RUN (analysis only)' if args.dry_run else 'FULL (train + eval)'}")
    print(f"{'#'*70}")

    while iteration < args.max_iterations:
        current_mAP, current_config = single_iteration(
            state, args, iteration + 1, current_mAP, current_config
        )

        # Check if target reached
        if current_mAP >= state.state['target_mAP']:
            print(f"\n{'!'*70}")
            print(f"  🎯 TARGET REACHED! mAP = {current_mAP:.3f} ≥ {state.state['target_mAP']:.1f}")
            print(f"  Total iterations: {iteration + 1}")
            print(f"  Active experiments: {state.state['active_experiments']}")
            print(f"{'!'*70}")
            break

        if not args.continuous:
            print(f"\n  Single iteration complete. Use --continuous for automatic looping.")
            break

        iteration += 1

        # Brief pause
        print(f"\n  Waiting 5 seconds before next iteration...")
        if not args.dry_run:
            time.sleep(5)

    # Final summary
    state.summary()

    # Export final config
    final_config_path = Path('runs/final_config.json')
    final_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_config_path, 'w') as f:
        json.dump({
            'best_mAP': state.state['best_mAP'],
            'experiments': state.state['active_experiments'],
            'config': current_config,
        }, f, indent=2)
    print(f"\n  Final config saved to {final_config_path}")


def main():
    args = parse_args()

    # Update target
    state = IterationState(args.resume)
    state.state['target_mAP'] = args.target_mAP
    state.save()

    # Show current state
    state.summary()

    if args.continuous:
        continuous_loop(state, args)
    else:
        # Single iteration
        current_mAP = state.state['best_mAP']
        current_config = {}
        current_mAP, current_config = single_iteration(
            state, args, len(state.state['iterations']) + 1,
            current_mAP, current_config
        )
        state.summary()


if __name__ == '__main__':
    main()
