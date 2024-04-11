from pathlib import Path
from typing import Sequence

from nneval.evaluate.semantic_eval import evaluate_semantic_results, get_samplewise_statistics
from nneval.utils.datastructures import SemanticResult
from nneval.utils.io import export_results, get_matching_semantic_pairs, save_json
from loguru import logger


def semantic_evaluation(
    semantic_pd_path: Path, semantic_gt_path: Path, output_path: Path, classes_of_interest: Sequence[int] = (1)
):
    """
    Evaluate the semantic results by comparing the predicted semantic labels with the ground truth labels.

    Args:
        semantic_pd_path (Path): The path to the predicted semantic labels.
        semantic_gt_path (Path): The path to the ground truth semantic labels.
        output_path (Path): The path to save the evaluation results.
        classes_of_interest (Sequence[int], optional): The classes of interest to evaluate. Defaults to (1).

    Returns:
        None
    """
    # ------------------------- Get all Cases to evaluate ------------------------ #
    logger.info("Get all matching semantic pairs to evaluate.")
    semantic_pairs = get_matching_semantic_pairs(gt_path=semantic_gt_path, pd_path=semantic_pd_path)
    logger.info(f"Found {len(semantic_pairs)} matching semantic pairs to evaluate.")
    # ----------- Evaluate Cases for all class ids and collect metrics ----------- #
    logger.info("Evaluating all semantic values.")
    eval: list[SemanticResult] = evaluate_semantic_results(semantic_pairs, classes_of_interest)
    # ------------------------- Save the results ------------------------- #
    output_path.mkdir(parents=True, exist_ok=True)
    export_results(eval, output_path)
    aggregated_results = get_samplewise_statistics(eval)
    save_json(aggregated_results, output_path / "aggregated_results.json")


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
