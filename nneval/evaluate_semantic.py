from pathlib import Path
from typing import Sequence

from loguru import logger
from nneval.evaluate.semantic_eval import evaluate_semantic_results
from nneval.evaluate.semantic_eval import get_samplewise_statistics
from nneval.utils.datastructures import SemanticResult
from nneval.utils.io import export_results
from nneval.utils.io import get_default_output_path
from nneval.utils.io import get_matching_semantic_pairs
from nneval.utils.io import save_json


def semantic_evaluation(
    semantic_pd_path: Path | str,
    semantic_gt_path: str | Path,
    output_path: str | Path,
    classes_of_interest: Sequence[int | Sequence[int]] = (1),
    save_to_disk: bool = True,
    output_name: str | None = None,
    n_processes: int = 1,
) -> tuple[dict, list[SemanticResult]]:
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

    semantic_pd_path = Path(semantic_pd_path)
    semantic_gt_path = Path(semantic_gt_path)
    output_path = Path(output_path)

    # ------------------------- Get all Cases to evaluate ------------------------ #
    logger.info("Get all matching semantic pairs to evaluate.")
    semantic_pairs = get_matching_semantic_pairs(gt_path=semantic_gt_path, pd_path=semantic_pd_path)
    logger.info(f"Found {len(semantic_pairs)} matching semantic pairs to evaluate.")
    # ----------- Evaluate Cases for all class ids and collect metrics ----------- #
    logger.info("Evaluating all semantic values.")
    eval: list[SemanticResult] = evaluate_semantic_results(
        semantic_pairs, classes_of_interest, n_processes=n_processes
    )
    # ------------------------- Save the results ------------------------- #
    output_path.mkdir(parents=True, exist_ok=True)
    export_results(eval, output_path, output_name)
    aggregated_results = get_samplewise_statistics(eval, classes_of_interest)
    if save_to_disk:
        if output_name is None:
            save_json(aggregated_results, output_path / "semantic_results_aggregated.json")
        else:
            save_json(aggregated_results, output_path / (output_name + "_agg.json"))
    return aggregated_results, eval


if __name__ == "__main__":
    cur_path = Path(__file__).parent.parent / "tests/"

    instance_pd_path = Path("/home/tassilowald/Data/aims-tbi/postprocess_eval/predsTr_5fold")
    instance_gt_path = Path("/home/tassilowald/Data/aims-tbi/postprocess_eval/labelsTr")
    out_gt_path = Path("/home/tassilowald/Data/aims-tbi/postprocess_eval/eval_results")

    semantic_evaluation(
        semantic_pd_path=instance_pd_path,
        semantic_gt_path=instance_gt_path,
        output_path=out_gt_path,
        classes_of_interest=(1,),
    )

    # semantic_evaluation(
    #     semantic_pd_path=multi_cls_pds,
    #     semantic_gt_path=multi_cls_gts,
    #     output_path=multi_out,
    #     classes_of_interest=(1, 2),
    # )
