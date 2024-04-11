import argparse
from pathlib import Path

from nneval.generate_instances import instance_generation


def create_instances_entrypoint():
    parser = argparse.ArgumentParser(description="Generate instances.")
    parser.add_argument("--semantic_pd_path", type=str, help="Path to semantic predictions.")
    parser.add_argument("--semantic_gt_path", type=str, help="Path to semantic ground truth.")
    parser.add_argument("--output_pd_path", type=str, help="Path to output predictions.")
    parser.add_argument("--output_gt_path", type=str, help="Path to output ground truth.")
    args = parser.parse_args()

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
