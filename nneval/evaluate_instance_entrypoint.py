from argparse import ArgumentParser
from pathlib import Path

from nneval.evaluate_instance import instance_evaluation


def instance_evaluation_entrypoint():
    parser = ArgumentParser()
    parser.add_argument(
        "-pd",
        "--instance_pd_path",
        type=Path,
        required=True,
        help="Path to instanced predictions",
    )
    parser.add_argument(
        "-gt",
        "--instance_gt_path",
        type=Path,
        required=True,
        help="Path to instanced groundtruths",
    )
    parser.add_argument(
        "-cls",
        "--classes_of_interest",
        type=int,
        nargs="+",
        default=(1,),
        help="Classes to evaluate",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Output path for evaluation results -- Uses prediction path if not provided",
    )
    args = parser.parse_args()

    pd_path: Path
    gt_path: Path
    output_path: Path
    pd_path = args.instance_pd_path
    gt_path = args.instance_gt_path
    output_path = args.output_path

    if not pd_path.is_absolute():
        pd_path = Path.cwd() / pd_path
    if not gt_path.is_absolute():
        gt_path = Path.cwd() / gt_path
    if not output_path.is_absolute():
        output_path = pd_path.parent / output_path

    output_path.mkdir(parents=True, exist_ok=True)

    classes_of_interest = args.classes_of_interest

    assert pd_path.exists(), f"Path to predictions does not exist: {pd_path}"
    assert gt_path.exists(), f"Path to groundtruth does not exist: {gt_path}"
    assert all([isinstance(c, int) for c in classes_of_interest]), "Classes of interest must be integers"
    instance_evaluation(pd_path, gt_path, output_path, classes_of_interest)


if __name__ == "__main__":
    instance_evaluation_entrypoint()
