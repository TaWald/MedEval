from pathlib import Path
from typing import Sequence
from nneval.evaluate.semantic_eval import evaluate_semantic_results

from nneval.utils.datastructures import SemanticResult
import pandas as pd
from nneval.utils.io import export_results, get_matching_semantic_pairs
from nneval.exporting import export_semantic


def semantic_evaluation(
    semantic_pd_path: Path, semantic_gt_path: Path, output_path: Path, classes_of_interest: Sequence[int] = (1)
):
    # ------------------------- Get all Cases to evaluate ------------------------ #
    semantic_pairs = get_matching_semantic_pairs(gt_path=semantic_gt_path, pd_path=semantic_pd_path)
    # ----------- Evaluate Cases for all class ids and collect metrics ----------- #
    eval: list[SemanticResult] = evaluate_semantic_results(semantic_pairs, classes_of_interest)
    # ------------------------- Save the results ------------------------- #
    export_results(eval, output_path)
    pass


if __name__ == "__main__":
    cur_path = Path(__file__).parent.parent / "tests/"

    single_cls_gts = cur_path / "sem_test_data/single_class/gts"
    single_cls_pds = cur_path / "sem_test_data/single_class/preds"
    multi_cls_gts = cur_path / "sem_test_data/multi_class/gts"
    multi_cls_pds = cur_path / "sem_test_data/multi_class/preds"

    single_out = cur_path / "sem_test_data/single_class/eval"
    multi_out = cur_path / "sem_test_data/multi_class/eval"

    semantic_evaluation(
        semantic_pd_path=single_cls_pds,
        semantic_gt_path=single_cls_gts,
        output_path=single_out,
        classes_of_interest=(1,),
    )

    semantic_evaluation(
        semantic_pd_path=multi_cls_pds,
        semantic_gt_path=multi_cls_gts,
        output_path=multi_out,
        classes_of_interest=(1, 2),
    )
