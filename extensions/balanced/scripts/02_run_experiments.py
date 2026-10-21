"""手动启动 pilot/full；--profile smoke 只使用合成数据。"""
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
    ap.add_argument('--experiments', default=None, help='逗号分隔实验名；自动补静态 LR 与必要参考组')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--reuse-from', default=None, help='显式导入已审查的原缓存签名；校验数据/参数/环境和每折索引')
    ap.add_argument('--check-only', action='store_true', help='只检查数据、旧缓存及实验列表，不训练或写结果')
    args = ap.parse_args()
    options = profile_options(args.profile)
    if args.jobs is not None:
        if args.jobs < 1:
            ap.error('--jobs 必须 >=1')
        options['jobs'] = args.jobs
    names = [x.strip() for x in args.experiments.split(',') if x.strip()] if args.experiments else list(CONFIG['experiments'])
    run_evaluation(options, names, reuse_from=args.reuse_from, check_only=args.check_only)


if __name__ == '__main__':
    main()
