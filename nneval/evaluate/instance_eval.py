from copy import deepcopy
from functools import partial
from itertools import chain, groupby
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from dataclasses import fields

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from toinstance import InstanceNrrd

from nneval.utils.datastructures import Instance, PredGTPair, InstanceResult, LesionwiseCasewiseResult
from nneval.utils.default_values import (
    no_groundtruth_but_prediction,
    no_prediction_but_groundtruth,
    no_prediction_no_groundtruth,
)


def aggregate_lesion_wise_case_wise_metrics(vals: dict[int, list[InstanceResult]]) -> dict:
    """Aggregates the results across all"""

    aggregated_results: dict[int, dict[str, float]] = {}

    for sem_id, results in vals.items():
        numeric_keys = [k.name for k in fields(results[0]) if isinstance(getattr(results[0], k.name), (float, int))]
        # Only use values of the same semantic class.
        sem_class_res = {}
        for k in numeric_keys:
            sem_class_res[k + "_mean"] = float(np.nanmean([getattr(v, k) for v in results]))
            sem_class_res[k + "_median"] = float(np.nanmedian([getattr(v, k) for v in results]))
            sem_class_res[k + "_q1"] = float(np.nanquantile([getattr(v, k) for v in results], 0.25))
            sem_class_res[k + "_q3"] = float(np.nanquantile([getattr(v, k) for v in results], 0.75))
        aggregated_results[sem_id] = sem_class_res
    return aggregated_results


def lesion_wise_case_wise_averaging(vals: Iterable[InstanceResult]):
    # This is also debateable: Not matched instances & groundtruths count half with their zeros.
    #   This is since they have no partner, a lesion that has a really low overlap would be worse
    #   Otherwise matching two instances with 0 would be better than not "matched" instances
    #   Properly dealing with this is not trivial, unfortunately
    vals = list(vals)
    lw_prec = float(np.nanmean([v.precision for v in vals]))
    lw_rec = float(np.nanmean([v.recall for v in vals]))
    lw_dice = float(np.nanmean([v.dice for v in vals]))
    if np.isnan(lw_prec) or np.isnan(lw_rec) or np.isnan(lw_dice):
        print("Wait")
    tps = np.nansum([v.true_positive for v in vals])
    fns = np.nansum([v.false_negative for v in vals])
    fps = np.nansum([1 - v.true_positive for v in vals])

    n_gts = sum([not np.isnan(v.false_negative) for v in vals])
    n_pds = sum([not np.isnan(v.true_positive) for v in vals])

    # ----------------------------- Case wise metrics ---------------------------- #
    # We also calculate case-wise metrics here, as they can be affected by lesion-wise filtering
    cw_intersect_voxels = float(np.nansum([v.intersection_voxels for v in vals]))
    cw_union_voxels = float(np.nansum([v.union_voxels for v in vals]))
    if cw_union_voxels == 0:
        cw_dice = 1
    elif cw_intersect_voxels == 0:
        cw_dice = 0
    else:
        cw_dice = 2 * cw_intersect_voxels / (cw_intersect_voxels + cw_union_voxels)

    # This has to be debated how to best handle the different cases!
    #   Case without any GT instance and no prediction instance): Precision: 1, Recall: 1, F1: 1
    #   Case without any GT instance and prediction instance: Precision: 0, Recall: 0, F1: 0
    #   Case with GT instance and no prediction instance: Precision: 0, Recall: 0, F1: 0
    if n_gts == 0 and n_pds == 0:
        case_precision = 1
        case_recall = 1
        case_f1 = 1
    elif n_gts == 0 and n_pds > 0:
        case_precision = 0
        case_recall = 1
        case_f1 = 0
    elif n_gts > 0 and n_pds == 0:
        case_precision = 1
        case_recall = 0
        case_f1 = 0
    else:
        case_precision = tps / (tps + fps)
        case_recall = tps / (tps + fns)
    if (case_precision + case_recall) == 0:
        case_f1 = 0
    else:
        case_f1 = 2 * (case_precision * case_recall) / (case_precision + case_recall)
    return LesionwiseCasewiseResult(
        case_id=vals[0].case_id,
        semantic_class_id=vals[0].semantic_class_id,
        volume_per_voxel=vals[0].volume_per_voxel,
        lw_precision=lw_prec,
        lw_recall=lw_rec,
        lw_dice=lw_dice,
        cw_dice=cw_dice,
        case_precision=case_precision,
        case_recall=case_recall,
        case_f1=case_f1,
        dice_threshold=vals[0].dice_threshold,
        predicted_lesion_count=n_pds,
        groundtruth_lesion_count=n_gts,
        total_predicted_voxels=float(np.nansum([v.pd_voxels for v in vals])),
        total_groundtruth_voxels=float(np.nansum([v.gt_voxels for v in vals])),
    )


def compare_all_instance_of_same_class(
    predictions: list[Instance],
    groundtruths: list[Instance],
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
        current_gt = cur_gt.bin_map
        for pd_idx, cur_pd in enumerate(predictions):
            current_pd = cur_pd.bin_map
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
    volume_threshold: float,
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
        "volume_threshold": volume_threshold,
    }
    # -------------------------- Create padded arrays ------------------------- #
    if len(predictions) == 0 and len(groundtruths) == 0:
        inst_result = no_prediction_no_groundtruth(**default_kwargs)
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
            pred: Instance = predictions[row]
            gt: Instance = groundtruths[col]

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
    instance_pair: PredGTPair,
    dice_threshold: float = 0.1,
    volume_threshold: float = 0,
    semantic_classes: Sequence[int] = (1,),
) -> list[InstanceResult]:
    """
    :param instance_pair: Contains paths to the prediction and groundtruth semantic and instance maps.
    :param dice_threshold: Set threshold determining if a prediction is considered a true positive.
    :param semantic_classes: List of semantic classes to evaluate.
    :return: List of matched predictions and List of matched groundtruths
    """

    instance_pd: InstanceNrrd = InstanceNrrd.from_innrrd(instance_pair.pd_p)
    instance_gt: InstanceNrrd = InstanceNrrd.from_innrrd(instance_pair.gt_p)

    ins_pd_arr: dict[int, list[np.ndarray]] = {int(k): v for k, v in instance_pd.get_semantic_instance_maps().items()}
    ins_gt_arr: dict[int, list[np.ndarray]] = {int(k): v for k, v in instance_gt.get_semantic_instance_maps().items()}

    spacing = instance_pd.get_spacing()
    dimensions = instance_pd.get_size()

    sample_name: str = instance_pair.pd_p.name.split(".")[0]

    # read images, discard where both images are background (most of the image)
    # ~100x Speedup by doing this (will depend on sparsity)

    # If neither gt nor pd values exist for this sample.
    all_matches: list[InstanceResult] = []
    for cls_id in semantic_classes:
        if cls_id not in ins_pd_arr:
            pd_instances = []
        else:
            pd_instances: list[np.ndarray] = ins_pd_arr[cls_id]
        if cls_id not in ins_gt_arr:
            gt_instances = []
        else:
            gt_instances: list[np.ndarray] = ins_gt_arr[cls_id]

        # We remove all voxels that are zero for all groundtruths and predictions as they don't matter.
        if len(pd_instances) == 0 and len(gt_instances) == 0:
            pass
        elif len(pd_instances) == 0:
            sem_area_rel = np.sum(np.stack(gt_instances, axis=0), axis=0) != 0
        elif len(gt_instances) == 0:
            sem_area_rel = np.sum(np.stack(pd_instances, axis=0), axis=0) != 0
        else:
            sem_area_rel = (np.sum(np.stack(pd_instances, axis=0), axis=0) != 0) | (
                np.sum(np.stack(gt_instances, axis=0), axis=0) != 0
            )

        predictions: list[Instance] = []
        for cnt, pd_inst in enumerate(pd_instances):
            relevant_arr = pd_inst[sem_area_rel]
            voxels = np.sum(relevant_arr)
            predictions.append(Instance(index=cnt, voxels=voxels, bin_map=relevant_arr))

        predictions = [pred for pred in predictions if (pred.voxels * np.prod(spacing)) >= volume_threshold]

        groundtruths: list[Instance] = []
        for cnt, gt_inst in enumerate(gt_instances):
            relevant_arr = gt_inst[sem_area_rel]
            voxels = np.sum(relevant_arr)
            groundtruths.append(Instance(index=cnt, voxels=voxels, bin_map=relevant_arr))

        all_to_all_comp = compare_all_instance_of_same_class(
            predictions=predictions,
            groundtruths=groundtruths,
        )
        one2one_matches: list[InstanceResult] = match_instances(
            predictions=predictions,
            groundtruths=groundtruths,
            dice_threshold=dice_threshold,
            volume_threshold=volume_threshold,
            spacing=spacing,
            dimensions=dimensions,
            all_to_all_comps=all_to_all_comp,
            class_id=cls_id,
            sample_name=sample_name,
        )
        all_matches.extend(deepcopy(one2one_matches))

    return all_matches


def evaluate_instance_results(
    instance_pair: list[PredGTPair],
    dice_threshold: float = 0.1,
    volume_threshold: float = 0,
    semantic_classes: Sequence[int] = (1,),
    num_processes: int = 1,
) -> List[InstanceResult]:
    eval_instance = partial(
        evaluate_instance_result,
        dice_threshold=dice_threshold,
        semantic_classes=semantic_classes,
        volume_threshold=volume_threshold,
    )
    if num_processes > 1:
        results = process_map(eval_instance, instance_pair, max_workers=num_processes)
    else:
        results = [eval_instance(ip) for ip in tqdm(instance_pair, desc="Evaluating Cases...")]

    return list(chain(*results))  # unnest


def get_samplewise_instance_wise_statistics(
    samplewise_results: list[InstanceResult],
    classes_of_interest: Sequence[int] = (1,),
) -> dict[int, List[InstanceResult]]:
    """
    Calculate sample-wise statistics for semantic evaluation.

    Args:
        samplewise_results (list[SemanticResult]): A list of SemanticResult objects containing the evaluation results for each sample.
        classes_of_interest (Sequence[int], optional): A sequence of class IDs for which to calculate the statistics. Defaults to (1,).

    Returns:
        dict: A dictionary containing the sample-wise statistics for each class of interest.

    """
    all_results = {}
    coi: InstanceResult
    for coi in classes_of_interest:
        res_oi = [entry for entry in samplewise_results if entry.semantic_class_id == coi]
        res_oi = groupby(res_oi, key=lambda x: x.case_id)
        all_results[coi] = [lesion_wise_case_wise_averaging(case_res[1]) for case_res in res_oi]

    return all_results


if __name__ == "__main__":
    some_pd_path = ""
    some_gt_path = ""
