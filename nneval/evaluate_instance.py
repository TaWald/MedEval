from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence
from nneval.evaluate.instance_eval import evaluate_instance_results

from nneval.utils.datastructures import InstanceResult
import pandas as pd


from nneval.utils.io import export_results, get_matching_instance_pairs


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
        required=False,
        default=None,
        help="Output path for evaluation results -- Uses prediction path if not provided",
    )
    args = parser.parse_args()

    if args.output_path is None:
        output_path = args.instance_pd_path
    else:
        output_path = args.output_path
    pd_path = args.instance_pd_path
    gt_path = args.instance_gt_path
    classes_of_interest = args.classes_of_interest

    assert pd_path.exists(), f"Path to predictions does not exist: {pd_path}"
    assert gt_path.exists(), f"Path to groundtruth does not exist: {gt_path}"
    assert all([isinstance(c, int) for c in classes_of_interest]), "Classes of interest must be integers"
    instance_evaluation(pd_path, gt_path, output_path, classes_of_interest)


def instance_evaluation(
    instance_pd_path: Path,
    instance_gt_path: Path,
    output_path: Path,
    classes_of_interest: Sequence[int] = (1),
    dice_threshold=0.1,
):
    # ---------------------------- Get instance cases ---------------------------- #
    instance_pairs = get_matching_instance_pairs(gt_path=instance_gt_path, pd_path=instance_pd_path)
    # ----------- Evaluate Cases for all class ids and collect metrics ----------- #
    eval: list[InstanceResult] = evaluate_instance_results(
        instance_pair=instance_pairs,
        dice_threshold=dice_threshold,
        semantic_classes=classes_of_interest,
    )
    # ------------------------- Save the results ------------------------- #
    export_results(eval, output_path)


if __name__ == "__main__":
    cur_path = Path(__file__).parent.parent / "tests/"

    single_cls_gts = cur_path / "ins_test_data/single_class/gts"
    single_cls_pds = cur_path / "ins_test_data/single_class/preds"
    multi_cls_gts = cur_path / "ins_test_data/multi_class/gts"
    multi_cls_pds = cur_path / "ins_test_data/multi_class/preds"

    single_out = cur_path / "ins_test_data/single_class/eval"
    multi_out = cur_path / "ins_test_data/multi_class/eval"

    # instance_evaluation(
    #     instance_pd_path=single_cls_pds,
    #     instance_gt_path=single_cls_gts,
    #     output_path=single_out,
    #     classes_of_interest=(1,),
    # )

    instance_evaluation(
        instance_pd_path=multi_cls_pds,
        instance_gt_path=multi_cls_gts,
        output_path=multi_out,
        classes_of_interest=(1, 2),
    )
