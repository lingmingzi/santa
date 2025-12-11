import argparse
import sys
from pathlib import Path

from runner import optimize_pipeline
from validation import score_and_validate_submission


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / 'sample_submission.csv'
    default_output = base_dir / 'submission.csv'

    parser = argparse.ArgumentParser(description="GPU/CPU baseline optimizer and validator")
    parser.add_argument('--mode', choices=['optimize', 'validate'], default='optimize',
                        help="Run optimization or validate an existing submission")
    parser.add_argument('--input', default=str(default_input),
                        help="Input submission CSV (default uses baseline/sample_submission.csv)")
    parser.add_argument('--output', default=str(default_output), help="Output CSV for optimized results")
    parser.add_argument('--limit', type=int, nargs='*', default=None,
                        help="Optional list of N values to process (optimize mode)")
    parser.add_argument('--iters', type=int, default=20, help="Simulated annealing iterations (optimize mode)")
    parser.add_argument('--restarts', type=int, default=10, help="Restart count (optimize mode)")
    parser.add_argument('--max-n', type=int, default=200, help="Maximum N to validate (validate mode)")
    args = parser.parse_args()
    if args.mode == 'optimize':
        optimize_pipeline(limit_n=args.limit, input_file=args.input, output_file=args.output,
                          iters=args.iters, restarts=args.restarts)
    else:
        result = score_and_validate_submission(args.input, max_n=args.max_n)
        print(result)


if __name__ == '__main__':
    main()
