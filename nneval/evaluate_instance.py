from pathlib import Path
from typing import Sequence
from nneval.evaluate.instance_eval import evaluate_instance_results
from nneval.utils.datastructures import InstanceResult
from nneval.utils.io import export_results, get_matching_instance_pairs


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
