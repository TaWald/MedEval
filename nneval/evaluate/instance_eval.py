from copy import deepcopy
from functools import partial
from itertools import chain, repeat
from pydicom import Sequence
from tqdm.contrib.concurrent import process_map  # or thread_map


from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa
import SimpleITK as sitk

from nneval.utils.datastructures import Instance, InstancePair, InstanceResult
from nneval.utils.default_values import (
    no_groundtruth_but_prediction,
    no_prediction_but_groundtruth,
    no_prediction_no_groundtruth,
)
from nneval.utils.io import get_all_sample_names, get_spacing_from_image


def compare_all_instance_of_same_class(
    predictions: list[Instance],
    groundtruths: list[Instance],
    flat_pd_npy: np.ndarray,
    flat_gt_npy: np.ndarray,
):
    """
    Compares all prediction to all groundtruth instances.
    Determines Dice, Precision, Recall, Intersection Vol., Union Vol.
    for all possible matches
    :param predictions: List of prediction instances -- Contains size, index and value
    (This can be non-consecutive when filtering by volume --> not always index == value)
    :param groundtruths: List of groundtruth instances -- Same content as predictions
    :param flat_pd_npy: Array with values of pd where either pd or gt are non zero
    :param flat_gt_npy: Array with values of gt where either pd or gt are non zero
    """

    n_pds = len(predictions)
    n_gts = len(groundtruths)

    dice_overlaps = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_iou = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_precisions = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_recalls = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_intersections = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_unions = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    prediction_index_id = np.zeros(shape=(n_pds, n_gts), dtype=np.int32)
    groundtruth_index_id = np.zeros(shape=(n_pds, n_gts), dtype=np.int32)

    for gt_idx, cur_gt in enumerate(groundtruths):
        # Actually accessing this one by one is faster than doing it in a big
        # batch, because why the fuck not ??!?!??!
        current_gt = flat_gt_npy == cur_gt.index
        for pd_idx, cur_pd in enumerate(predictions):
            current_pd = flat_pd_npy == cur_pd.index
            union = np.count_nonzero(current_gt | current_pd)
            intersect = np.count_nonzero(current_gt & current_pd)
            precision = intersect / cur_pd.voxels
            recall = intersect / cur_gt.voxels

            dice = (2 * intersect) / (intersect + union)
            iou = intersect / union
            dice_overlaps[pd_idx, gt_idx] = dice
            all_iou[pd_idx, gt_idx] = iou
            all_precisions[pd_idx, gt_idx] = precision
            all_recalls[pd_idx, gt_idx] = recall
            all_intersections[pd_idx, gt_idx] = intersect
            all_unions[pd_idx, gt_idx] = union
            prediction_index_id[pd_idx, gt_idx] = pd_idx
            groundtruth_index_id[pd_idx, gt_idx] = gt_idx
    return {
        "dice": dice_overlaps,
        "iou": all_iou,
        "precision": all_precisions,
        "recall": all_recalls,
        "intersection": all_intersections,
        "union": all_unions,
        "pd_instance_id": prediction_index_id,
        "gt_instance_id": groundtruth_index_id,
    }


def match_instances(
    predictions: list[Instance],
    groundtruths: list[Instance],
    dice_threshold: float,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    all_to_all_comps: Dict,
    sample_name: str,
    class_id: int,
) -> list[InstanceResult]:
    results: list[InstanceResult] = []

    vol_per_voxel = float(np.prod(spacing))
    # Prediction in rows and GTs in Cols.
    dice_arr = all_to_all_comps["dice"]
    iou_arr = all_to_all_comps["iou"]
    prec_arr = all_to_all_comps["precision"]
    reca_arr = all_to_all_comps["recall"]
    inte_arr = all_to_all_comps["intersection"]
    unio_arr = all_to_all_comps["union"]

    default_kwargs = {
        "case_id": sample_name,
        "semantic_class_id": class_id,
        "volume_per_voxel": vol_per_voxel,
        "spacing": spacing,
        "dimensions": dimensions,
        "dice_threshold": dice_threshold,
    }
    # -------------------------- Create padded arrays ------------------------- #
    if len(predictions) == 0 and len(groundtruths) == 0:
        inst_result = no_prediction_no_groundtruth(dice_threshold=dice_threshold, **default_kwargs)
        results.append(inst_result)
    elif len(predictions) == 0:
        for gt in groundtruths:
            inst_result = no_prediction_but_groundtruth(gt=gt, **default_kwargs)
            results.append(inst_result)
    elif len(groundtruths) == 0:
        for pred in predictions:
            inst_result = no_groundtruth_but_prediction(pred=pred, **default_kwargs)
            results.append(inst_result)
    else:
        row_ids, col_ids = lsa(-dice_arr)  # Finds Hungarian matching solution maximizing dice

        set_of_predictions = set(np.arange(len(predictions)))
        set_of_groundtruths = set(np.arange(len(groundtruths)))

        not_matched_preds = set_of_predictions - set(row_ids)
        not_matched_gts = set_of_groundtruths - set(col_ids)

        for row, col in zip(row_ids, col_ids):
            pred = predictions[row]
            gt = groundtruths[col]

            dice = float(dice_arr[row, col])
            if dice == 0:
                # If dice is zero we consider this not matched instances.
                not_matched_preds.add(row)
                not_matched_gts.add(col)
                continue
            # ------------------ Assign metrics for all whose dice != 0 ------------------ #
            inst_result = InstanceResult(
                dice=dice,
                iou=float(iou_arr[row, col]),
                precision=float(prec_arr[row, col]),
                recall=float(reca_arr[row, col]),
                pd_voxels=pred.voxels,
                gt_voxels=gt.voxels,
                union_voxels=unio_arr[row, col],
                intersection_voxels=inte_arr[row, col],
                pd_instance_index=pred.index,
                gt_instance_index=gt.index,
                match=True,
                **default_kwargs,
            )
            results.append(inst_result)
        # ------------ Set all with Dice 0 to not default undefined values ----------- #
        for nmp in not_matched_preds:
            pred = predictions[nmp]
            res_dict = no_groundtruth_but_prediction(pred=pred, **default_kwargs)
            results.append(res_dict)
        for nmg in not_matched_gts:
            gt = groundtruths[nmg]
            res_dict = no_prediction_but_groundtruth(gt=gt, **default_kwargs)
            results.append(res_dict)
    return results


def instance_id_to_semantic_class_id(semantic_map: np.ndarray, instance_map: np.ndarray):
    """
    Determines the mapping of instances to semantic classes.
    Needed so only the instances of the same semantic class are matched.
    :param semantic_map: Semantic segmentation map
    :param instance_map: Instance segmentation map
    """
    non_zero_instances = np.unique(instance_map[instance_map != 0])
    mapping = {}
    for nzi in non_zero_instances:
        sem_cls = semantic_map[np.argwhere(instance_map == nzi)[0]]
        mapping[nzi] = sem_cls
    return mapping


def evaluate_instance_result(
    instance_pair: InstancePair,
    dice_threshold: float = 0.1,
    semantic_classes: Sequence[int] = (1,),
) -> list[InstanceResult]:
    """
    :param instance_pair: Contains paths to the prediction and groundtruth semantic and instance maps.
    :param dice_threshold: Set threshold determining if a prediction is considered a true positive.
    :param semantic_classes: List of semantic classes to evaluate.
    :return: List of matched predictions and List of matched groundtruths
    """

    semantic_pd: sitk.Image = sitk.ReadImage(str(instance_pair.semantic_pd_p))
    semantic_gt: sitk.Image = sitk.ReadImage(str(instance_pair.semantic_gt_p))
    instance_pd: sitk.Image = sitk.ReadImage(str(instance_pair.instance_pd_p))
    instance_gt: sitk.Image = sitk.ReadImage(str(instance_pair.instance_gt_p))

    sem_pd_arr: np.ndarray = sitk.GetArrayFromImage(semantic_pd)
    sem_gt_arr: np.ndarray = sitk.GetArrayFromImage(semantic_gt)
    ins_pd_arr: np.ndarray = sitk.GetArrayFromImage(instance_pd)
    ins_gt_arr: np.ndarray = sitk.GetArrayFromImage(instance_gt)

    spacing = semantic_pd.GetSpacing()
    dimensions = sem_pd_arr.shape

    sample_name: str = instance_pair.semantic_pd_p.name

    # read images, discard where both images are background (most of the image)
    # ~100x Speedup by doing this (will depend on sparsity)

    # If neither gt nor pd values exist for this sample.
    all_matches: list[InstanceResult] = []
    for cls_id in semantic_classes:
        semantic_area_relevant = (sem_pd_arr == cls_id) | (sem_gt_arr == cls_id)

        remaining_pd_instances = ins_pd_arr[semantic_area_relevant]
        remaining_gt_instances = ins_gt_arr[semantic_area_relevant]

        pd_instances = np.unique(remaining_pd_instances, return_counts=True)
        gt_instance = np.unique(remaining_gt_instances, return_counts=True)

        predictions: list[Instance] = [
            Instance(index=index, voxels=size) for index, size in zip(pd_instances[0], pd_instances[1])
        ]
        groundtruths: list[Instance] = [
            Instance(index=index, voxels=size) for index, size in zip(gt_instance[0], gt_instance[1])
        ]

        all_to_all_comp = compare_all_instance_of_same_class(
            predictions=predictions,
            groundtruths=groundtruths,
            flat_pd_npy=remaining_pd_instances,
            flat_gt_npy=remaining_gt_instances,
        )
        one2one_matches = match_instances(
            predictions=predictions,
            groundtruths=groundtruths,
            dice_threshold=dice_threshold,
            spacing=spacing,
            dimensions=dimensions,
            all_to_all_comps=all_to_all_comp,
            class_id=cls_id,
            sample_name=sample_name,
        )
        all_matches.extend(deepcopy(one2one_matches))

    return all_matches


def evaluate_instance_results(
    instance_pair: list[InstancePair],
    dice_threshold: float = 0.1,
    semantic_classes: Sequence[int] = (1,),
    num_processes: int = 1,
) -> List[InstanceResult]:
    eval_instance = partial(
        evaluate_instance_result, dice_threshold=dice_threshold, semantic_classes=semantic_classes
    )
    if num_processes > 1:
        results = process_map(eval_instance, instance_pair, max_workers=num_processes)
    else:
        results = [eval_instance(ip) for ip in instance_pair]

    return list(chain(*results))  # unnest


if __name__ == "__main__":
    some_pd_path = ""
    some_gt_path = ""
