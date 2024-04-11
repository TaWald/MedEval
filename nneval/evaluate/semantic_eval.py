from functools import partial
from itertools import chain
from multiprocessing import Pool
import os
from pathlib import Path
import sys
from typing import Sequence

import SimpleITK as sitk
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import norm

from nneval.utils.datastructures import SemanticPair, SemanticResult
from nneval.utils.io import get_array_from_image
from nneval.utils.parser import get_samplewise_eval_parser
from tqdm.contrib.concurrent import process_map  # or thread_map


def samplewise_eval(
    gt_array: np.ndarray,
    gt_class_of_interest: int,
    pd_array: np.ndarray,
    pd_class_of_interest: int,
    sample_name: str,
    vol_per_voxel: float = 1.0,
) -> dict:
    gt_flat = np.reshape(gt_array, newshape=[-1])
    pd_flat = np.reshape(pd_array, newshape=[-1])

    gt_bool_npy = gt_flat == gt_class_of_interest
    pd_bool_npy = pd_flat == pd_class_of_interest

    #### Has to remove the instances where the prediction instance volume is smaller
    # than the threshold (only noticeable for bigger filter thresholds)

    joined_npy = np.logical_or(gt_bool_npy, pd_bool_npy)  # Where either of them is not background!

    gt_bool_npy = gt_bool_npy[joined_npy]
    pd_bool_npy = pd_bool_npy[joined_npy]

    pd_voxels = np.count_nonzero(pd_bool_npy)
    gt_voxels = np.count_nonzero(gt_bool_npy)

    union_voxels = np.count_nonzero(joined_npy)
    intersection_voxels = np.count_nonzero(np.logical_and(gt_bool_npy, pd_bool_npy))
    if union_voxels != 0:
        dice = (2 * intersection_voxels) / (union_voxels + intersection_voxels)
        iou = intersection_voxels / union_voxels
    else:
        dice = np.NAN
        iou = np.NAN

    if gt_voxels != 0:
        sensitivity = intersection_voxels / gt_voxels
    else:
        sensitivity = np.NAN

    if pd_voxels != 0:
        precision = intersection_voxels / pd_voxels
    else:
        precision = np.NAN

    return dict(
        groundtruth_voxels=gt_voxels,
        prediction_voxels=pd_voxels,
        volume_per_voxel_mm3=vol_per_voxel,
        intersection_voxels=intersection_voxels,
        union_voxels=union_voxels,
        sensitivity=sensitivity,
        precision=precision,
        dice=dice,
        iou=iou,
        pd_segmentation_class=pd_class_of_interest,
        gt_segmentation_class=gt_class_of_interest,
        gt_volume_mm3=gt_voxels * vol_per_voxel,
        pd_volume_mm3=pd_voxels * vol_per_voxel,
        sample_name=sample_name,
    )


def confidence_intervals(vals, confidence_level=0.95) -> tuple:
    """
    Calculates the 95% CI for a given list of values.
    Especially sued for Sensitivity, PPV, F1-Scores.
    """
    z_score = norm.ppf((1 + confidence_level) / 2)
    non_nan_vals = [val for val in vals if not np.isnan(val)]
    N = len(non_nan_vals)
    mean = np.mean(non_nan_vals)
    SEM = np.std(non_nan_vals) / np.sqrt(N)
    bounds = (float(mean - z_score * SEM), float(mean + z_score * SEM))

    return bounds


def get_samplewise_statistics(
    samplewise_results: list[SemanticResult],
    classes_of_interest: Sequence[int] = (1,),
) -> dict:
    """
    Calculate sample-wise statistics for semantic evaluation.

    Args:
        samplewise_results (list[SemanticResult]): A list of SemanticResult objects containing the evaluation results for each sample.
        classes_of_interest (Sequence[int], optional): A sequence of class IDs for which to calculate the statistics. Defaults to (1,).

    Returns:
        dict: A dictionary containing the sample-wise statistics for each class of interest.

    """
    all_results = {}
    for coi in classes_of_interest:
        res_oi = [entry for entry in samplewise_results if entry.class_id == coi]
        all_results[str(coi)] = (
            {
                "dice": {
                    "mean": float(np.nanmean([entry.dice for entry in res_oi])),
                    "std": float(np.nanstd([entry.dice for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.dice for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.dice for entry in res_oi])[1],
                    "q1": float(np.nanquantile([entry.dice for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.dice for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.dice for entry in res_oi], 0.75)),
                },
                "iou": {
                    "mean": float(np.nanmean([entry.iou for entry in res_oi])),
                    "std": float(np.nanstd([entry.iou for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.iou for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.iou for entry in res_oi])[1],
                    "q1": float(np.nanquantile([entry.iou for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.iou for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.iou for entry in res_oi], 0.75)),
                },
                "gt_volume_mm3": {
                    "mean": float(np.nanmean([entry.gt_volume for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.gt_volume for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.gt_volume for entry in res_oi])[1],
                    "std": float(np.nanstd([entry.gt_volume for entry in res_oi])),
                    "q1": float(np.nanquantile([entry.gt_volume for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.gt_volume for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.gt_volume for entry in res_oi], 0.75)),
                },
                "pd_volume_mm3": {
                    "mean": float(np.nanmean([entry.pd_volume for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.pd_volume for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.pd_volume for entry in res_oi])[1],
                    "std": float(np.nanstd([entry.pd_volume for entry in res_oi])),
                    "q1": float(np.nanquantile([entry.pd_volume for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.pd_volume for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.pd_volume for entry in res_oi], 0.75)),
                },
                "sensitivity": {
                    "mean": float(np.nanmean([entry.recall for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.recall for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.recall for entry in res_oi])[1],
                    "std": float(np.nanstd([entry.recall for entry in res_oi])),
                    "q1": float(np.nanquantile([entry.recall for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.recall for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.recall for entry in res_oi], 0.75)),
                },
                "precision": {
                    "mean": float(np.nanmean([entry.precision for entry in res_oi])),
                    "lower_CI95": confidence_intervals([entry.precision for entry in res_oi])[0],
                    "upper_CI95": confidence_intervals([entry.precision for entry in res_oi])[1],
                    "std": float(np.nanstd([entry.precision for entry in res_oi])),
                    "q1": float(np.nanquantile([entry.precision for entry in res_oi], 0.25)),
                    "q2": float(np.nanquantile([entry.precision for entry in res_oi], 0.5)),
                    "q3": float(np.nanquantile([entry.precision for entry in res_oi], 0.75)),
                },
            },
        )

    return all_results


def evaluate_semantic_results(
    sem_pairs: list[SemanticPair],
    class_ids_to_evaluate: Sequence[int],
    n_processes: int = 1,
) -> list[SemanticResult]:
    logger.info("Starting unresampled casewise evaluation.")

    if n_processes == 1:
        samplewise_results = []
        for sem_pair in sem_pairs:
            samplewise_results.extend(
                semantic_evaluate_case_sempair(
                    semantic_pair=sem_pair,
                    class_of_interests=class_ids_to_evaluate,
                )
            )
    else:
        semantic_eval = partial(semantic_evaluate_case_sempair, class_of_interests=class_ids_to_evaluate)
        samplewise_results = list(chain(process_map(semantic_eval, sem_pairs, max_workers=n_processes)))
    return samplewise_results


def _semantic_classwise_eval(gt_arr: np.ndarray, pd_arr: np.ndarray, class_id: int) -> SemanticResult:
    """Calculates key semantic_metrics that will be saved into a large file later."""
    gt_flat = np.reshape(gt_arr, newshape=[-1])
    pd_flat = np.reshape(pd_arr, newshape=[-1])

    gt_bool_npy = gt_flat == class_id
    pd_bool_npy = pd_flat == class_id

    #### Has to remove the instances where the prediction instance volume is smaller
    # than the threshold (only noticeable for bigger filter thresholds)

    joined_npy = np.logical_or(gt_bool_npy, pd_bool_npy)  # Where either of them is not background!

    gt_bool_npy = gt_bool_npy[joined_npy]
    pd_bool_npy = pd_bool_npy[joined_npy]

    pd_voxels = np.count_nonzero(pd_bool_npy)
    gt_voxels = np.count_nonzero(gt_bool_npy)

    union_voxels = np.count_nonzero(joined_npy)
    intersection_voxels = np.count_nonzero(np.logical_and(gt_bool_npy, pd_bool_npy))
    if union_voxels != 0:
        dice = (2 * intersection_voxels) / (union_voxels + intersection_voxels)
        iou = intersection_voxels / union_voxels
    else:
        dice = np.NAN
        iou = np.NAN

    if gt_voxels != 0:
        recall = intersection_voxels / gt_voxels
    else:
        recall = np.NAN

    if pd_voxels != 0:
        precision = intersection_voxels / pd_voxels
    else:
        precision = np.NAN

    # ToDo: Add more metrics here. Currently it's only boring stuff like dice, iou, recall, precision
    #   Can have more metrics like NSD, ASD, etc.

    return SemanticResult(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        gt_voxels=gt_voxels,
        pd_voxels=pd_voxels,
        union_voxels=union_voxels,
        intersection_voxels=intersection_voxels,
        class_id=class_id,
    )


def semantic_evaluate_case_sempair(
    semantic_pair: SemanticPair, class_of_interests: Sequence[int] = (1,)
) -> list[SemanticResult]:
    """Additional interface to start semantic_evaluate_case with a SemanticPair object."""
    semantic_pd_path = semantic_pair.pd_p
    semantic_gt_path = semantic_pair.gt_p
    return semantic_evaluate_case(semantic_pd_path, semantic_gt_path, class_of_interests)


def semantic_evaluate_case(
    semantic_pd_path: Path,
    semantic_gt_path: Path,
    class_of_interests: Sequence[int] = (1,),
) -> list[SemanticResult]:
    """
    Evaluates a single case for all semantic classes and returns a dictionary containing all metrics for this case.
    Metrics contained are:
    `['DICE', 'IOU', 'Recall(Sensitivity)', 'Precision(PPV)', 'GT Voxels', 'PD Voxels', 'Intersection Voxels', 'Union Voxels', 'Class ID',
    'Volume per Voxel in mm³', 'Spacing', 'Dimensions', 'Case ID']`
    If neither PD or GT are predicted --> dice = NaN, iou = NaN,
    If GT is empty --> recall = NaN
    If PD is empty --> precision = NaN

    :param semantic_pd_path: Path to the prediction of the semantic segmentation
    :param semantic_gt_path: Path to the groundtruth of the semantic segmentation
    :param class_of_interests: List of classes that should be evaluated
    """

    gt_im = sitk.ReadImage(semantic_gt_path)
    pd_im = sitk.ReadImage(semantic_pd_path)

    gt_spacing = gt_im.GetSpacing()
    pd_spacing = pd_im.GetSpacing()
    case_id = semantic_gt_path.name

    assert np.all(np.isclose(gt_spacing, pd_spacing)), "Not equal spacing"

    vol_factor = float(np.prod(gt_spacing))  # in mm³
    gt_npy = sitk.GetArrayFromImage(gt_im)
    pd_npy = sitk.GetArrayFromImage(pd_im)

    all_results: list[SemanticResult] = []
    for class_id in class_of_interests:
        semantic_eval: SemanticResult = _semantic_classwise_eval(gt_npy, pd_npy, class_id)
        semantic_eval.add_volume_per_voxel(vol_factor)
        semantic_eval.case_id = case_id
        semantic_eval.spacing = gt_spacing
        semantic_eval.dimensions = gt_npy.shape
        all_results.append(semantic_eval)

    return all_results  # List of dicts


def samplewise_result(
    gt_sample_path: str,
    pd_sample_path: str,
    gt_class_of_interest: int,
    pd_class_of_interest: int,
) -> dict:
    """Calculates the dice of the niftis given in gt_sample_path and pd_sample_path.
    The dice is calcualted between the voxels that are of gt_class_of_interest and
    pd_class_of_interest.

    :param gt_class_of_interest: Class used to compare in groundtruth
    :param pd_class_of_interest: Class used to compare in predictions
    :param gt_sample_path: Path to the gt sample to calc dice on
    :param pd_sample_path: Path to the pd sample to calc dice on
    """

    sample_name = os.path.basename(gt_sample_path)

    gt_im = sitk.ReadImage(gt_sample_path)
    pd_im = sitk.ReadImage(pd_sample_path)

    gt_spacing = gt_im.GetSpacing()
    pd_spacing = pd_im.GetSpacing()

    assert np.all(np.isclose(gt_spacing, pd_spacing)), "Not equal spacing"

    vol_factor = float(np.prod(gt_spacing))  # in mm³

    gt_npy = get_array_from_image(gt_sample_path)
    pd_npy = get_array_from_image(pd_sample_path)

    return samplewise_eval(
        gt_array=gt_npy,
        gt_class_of_interest=gt_class_of_interest,
        pd_array=pd_npy,
        pd_class_of_interest=pd_class_of_interest,
        sample_name=sample_name,
        vol_per_voxel=vol_factor,
    )


if __name__ == "__main__":
    parser = get_samplewise_eval_parser()
    args = parser.parse_args()
    gt_path = args.groundtruth_path
    pd_path = args.prediction_path
    out_path = args.output_path
    cls_of_interest = args.classes_of_interest
    if out_path is None:
        out_path = pd_path

    results = []
    sorted_ids = get_matching_gt_and_preds(gt_path, pd_path)
    for patient in sorted_ids:
        for c in cls_of_interest:
            patient_result = {"patient_id": patient, "class_id": c}
            pat_gt_path = os.path.join(gt_path, patient)
            pat_pd_path = os.path.join(pd_path, patient)
            patient_result.update(
                **samplewise_result(
                    gt_sample_path=pat_gt_path,
                    pd_sample_path=pat_pd_path,
                    gt_class_of_interest=c,
                    pd_class_of_interest=c,
                )
            )
            results.append(patient_result)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_path, "samplewise_evaluation.csv"), index=False)
    sys.exit(0)
