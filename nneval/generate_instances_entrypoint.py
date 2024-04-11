import argparse
from pathlib import Path

from nneval.generate_instances import instance_generation


def create_instances_entrypoint():
    parser = argparse.ArgumentParser(description="Generate instances.")
    parser.add_argument("--semantic_pd_path", type=str, help="Path to semantic predictions.")
    parser.add_argument("--semantic_gt_path", type=str, help="Path to semantic ground truth.")
    parser.add_argument("--output_pd_path", type=str, help="Path to output predictions.", required=False)
    parser.add_argument("--output_gt_path", type=str, help="Path to output ground truth.", required=False)
    args = parser.parse_args()

    if args.output_pd_path is None:
        args.output_pd_path = args.semantic_pd_path
        args.output_pd_path = Path(args.output_pd_path) / "nneval_instances"
        args.output_pd_path.mkdir(parents=False, exist_ok=True)

    if args.output_gt_path is None:
        args.output_gt_path = args.semantic_gt_path
        args.output_gt_path = Path(args.output_gt_path) / "nneval_instances"
        args.output_gt_path.mkdir(parents=False, exist_ok=True)

    semantic_pd_path = Path(args.semantic_pd_path)
    semantic_gt_path = Path(args.semantic_gt_path)
    output_pd_path = Path(args.output_pd_path)
    output_gt_path = Path(args.output_gt_path)

    instance_generation(
        semantic_pd_path=semantic_pd_path,
        semantic_gt_path=semantic_gt_path,
        output_pd_path=output_pd_path,
        output_gt_path=output_gt_path,
    )
