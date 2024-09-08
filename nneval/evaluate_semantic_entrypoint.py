from argparse import ArgumentParser
from pathlib import Path

from nneval.evaluate_semantic import semantic_evaluation


def semantic_evaluation_entrypoint():
    """
    Entry point function for semantic evaluation.

    This function parses command line arguments, validates the paths, and calls the `semantic_evaluation` function
    to perform the evaluation.

    Command line arguments:
    --semantic_pd_path: Path to predictions
    --semantic_gt_path: Path to groundtruth
    --classes_of_interest: Classes to evaluate
    --output_path: Output path for evaluation results (optional)

    Returns:
    None
    """
    parser = ArgumentParser()
    parser.add_argument("-pd", "--semantic_pd_path", type=Path, required=True, help="Path to predictions")
    parser.add_argument("-gt", "--semantic_gt_path", type=Path, required=True, help="Path to groundtruth")
    parser.add_argument(
        "-cls", "--classes_of_interest", type=int, nargs="+", default=(1,), help="Classes to evaluate"
    )
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    args = parser.parse_args()

    # if args.output_path is None:
    #     output_path = args.semantic_pd_path
    #     output_path = Path(output_path) / "nneval"
    #     output_path.mkdir(parents=False, exist_ok=True)
    # else:
    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    pd_path = args.semantic_pd_path
    gt_path = args.semantic_gt_path
    classes_of_interest = args.classes_of_interest

    assert pd_path.exists(), f"Path to predictions does not exist: {pd_path}"
    assert gt_path.exists(), f"Path to groundtruth does not exist: {gt_path}"
    assert all([isinstance(c, int) for c in classes_of_interest]), "Classes of interest must be integers"
    semantic_evaluation(pd_path, gt_path, output_path, classes_of_interest)


if __name__ == "__main__":
    semantic_evaluation_entrypoint()
