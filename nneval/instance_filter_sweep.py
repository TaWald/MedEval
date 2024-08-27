from pathlib import Path
from typing import Sequence

from tqdm import tqdm
from nneval.evaluate.instance_eval import aggregate_lesion_wise_case_wise_metrics
from nneval.evaluate.instance_eval import evaluate_instance_results
from nneval.evaluate.instance_eval import get_samplewise_instance_wise_statistics

from nneval.utils.datastructures import InstanceResult
from nneval.utils.io import get_matching_instance_pairs, save_json
import numpy as np


def instance_sweep_evaluation(
    instance_pd_path: str | Path,
    instance_gt_path: str | Path,
    output_path: str | Path,
    classes_of_interest: Sequence[int] = (1,),
    dice_threshold=0.1,
    volume_sweep_threshold: Sequence[float] = (0, 100, 21),
):

    instance_pd_path = Path(instance_pd_path)
    instance_gt_path = Path(instance_gt_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # ---------------------------- Get instance cases ---------------------------- #
    instance_pairs = get_matching_instance_pairs(gt_path=instance_gt_path, pd_path=instance_pd_path)
    volume_threshold = np.linspace(*volume_sweep_threshold, endpoint=True)
    # -------------------------------- Start sweep ------------------------------- #
    for vol_thres in tqdm(volume_threshold, desc="Volume Threshold Sweep"):
        # ----------- Evaluate Cases for all class ids and collect metrics ----------- #
        eval: list[InstanceResult] = evaluate_instance_results(
            instance_pair=instance_pairs,
            dice_threshold=dice_threshold,
            volume_threshold=vol_thres,
            semantic_classes=classes_of_interest,
        )
        # Aggregate lesions within case and then patients across dataset.
        sample_wise_instance_wise_results: dict[int, list[InstanceResult]] = get_samplewise_instance_wise_statistics(
            eval
        )
        aggregated_lwcw = aggregate_lesion_wise_case_wise_metrics(sample_wise_instance_wise_results)
        save_json(aggregated_lwcw, output_path / f"agg_lwcw_result_volthres_{vol_thres:.01f}.json")


if __name__ == "__main__":
    cur_path = Path(__file__).parent.parent / "tests/"

    instance_sweep_evaluation(
        instance_pd_path="/home/tassilowald/Data/aims-tbi/postprocess_eval/predsTr_5fold_instanced",
        instance_gt_path="/home/tassilowald/Data/aims-tbi/postprocess_eval/labelsTr_instanced",
        output_path="/home/tassilowald/Data/aims-tbi/postprocess_eval/sweep",
        classes_of_interest=(1,),
    )
