"""Start pilot/full manually; --profile smoke uses synthetic data only."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dn_extensions.evaluation import run_evaluation
from dn_extensions.settings import CONFIG, profile_options


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=['pilot', 'full', 'smoke'], default='pilot')
    ap.add_argument('--experiments', default=None, help='comma-separated experiment names; static LR and required reference groups are added automatically')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--reuse-from', default=None, help='explicitly import a reviewed source cache signature; checks data/parameters/environment and per-fold indices')
    ap.add_argument('--check-only', action='store_true', help='only check the data, old cache and experiment list; no training or results written')
    args = ap.parse_args()
    options = profile_options(args.profile)
    if args.jobs is not None:
        if args.jobs < 1:
            ap.error('--jobs must be >=1')
        options['jobs'] = args.jobs
    names = [x.strip() for x in args.experiments.split(',') if x.strip()] if args.experiments else list(CONFIG['experiments'])
    run_evaluation(options, names, reuse_from=args.reuse_from, check_only=args.check_only)


if __name__ == '__main__':
    main()
