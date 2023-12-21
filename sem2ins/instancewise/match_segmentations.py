from itertools import chain, repeat
import os
from multiprocessing import Pool
from time import time
from typing import Dict, List, Tuple

import numpy as np
from batchgenerators.utilities import file_and_folder_operations as ff
import pandas as pd
from scipy.optimize import linear_sum_assignment as lsa

import sem2ins.utils.configuration as conf
from sem2ins.utils.default_values import get_empty_prediction, get_empty_groundtruth, get_empty_strict
from sem2ins.utils.loading import get_all_sample_names, get_spacing_from_image
from sem2ins.utils.loading import get_flat_array_from_image


def find_instances(flat_array: np.ndarray) -> List[Dict]:
    """
    Iterates over the unique values in the array, collects some infos on the
    instances that are of non-zero value.
    {Size: Total number of voxels of instance,
    value: The integer value of the instance
    index: Increasing number of instances.}
    Normally index should be value -1 but sometimes small
    instances can be filtered before.

    :param flat_array: An array of ints. Same int value represent same instance.
    """
    instances = []
    existing_labels = sorted(np.unique(flat_array))
    assert all([lbl >= 0 for lbl in existing_labels]), "No negative values expected!"
    for cnt, i in enumerate(existing_labels):  # Uniques not consecutive? -> enum
        if i == 0:
            continue  # Skip background
        size = np.count_nonzero(flat_array == i)
        index = cnt - 1 if 0 in existing_labels else cnt
        instances.append({"size": size, "value": i, "index": index})
    return instances


def compare_all_pd_to_gt(
    predictions: List[Dict],
    groundtruths: List[Dict],
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
    all_precisions = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_recalls = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_intersections = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)
    all_unions = np.zeros(shape=(n_pds, n_gts), dtype=np.float32)

    for cur_gt in groundtruths:
        # Actually accessing this one by one is faster than doing it in a big
        # batch, because why the fuck not ??!?!??!
        current_gt = flat_gt_npy == cur_gt["value"]
        for cur_pd in predictions:
            current_pd = flat_pd_npy == cur_pd["value"]
            union = np.count_nonzero(current_gt | current_pd)
            intersect = np.count_nonzero(current_gt & current_pd)
            precision = intersect / cur_pd["size"]
            recall = intersect / cur_gt["size"]

            dice = (2 * intersect) / (intersect + union)
            dice_overlaps[cur_pd["index"], cur_gt["index"]] = dice
            all_precisions[cur_pd["index"], cur_gt["index"]] = precision
            all_recalls[cur_pd["index"], cur_gt["index"]] = recall
            all_intersections[cur_pd["index"], cur_gt["index"]] = intersect
            all_unions[cur_pd["index"], cur_gt["index"]] = union
    return {
        "dice": dice_overlaps,
        "precision": all_precisions,
        "recall": all_recalls,
        "intersection": all_intersections,
        "union": all_unions,
    }


def match_predictions(
    predictions: List[Dict],
    groundtruths: List[Dict],
    spacing: np.ndarray,
    all_to_all_comps: Dict,
    sample_name: str,
) -> List[Dict]:
    matched_predictions = []
    dice_arr = all_to_all_comps["dice"]
    prec_arr = all_to_all_comps["precision"]
    reca_arr = all_to_all_comps["recall"]
    inte_arr = all_to_all_comps["intersection"]
    unio_arr = all_to_all_comps["union"]
    vol_per_voxel = float(np.prod(spacing))

    for prediction in predictions:
        res_dict = dict()
        res_dict["type"] = "prediction"
        res_dict["image_name"] = sample_name
        # Info about the current prediction
        res_dict["prediction_size"] = prediction["size"]
        res_dict["prediction_index"] = prediction["index"]
        res_dict["volume_per_voxel_mm3"] = vol_per_voxel
        res_dict["spacing"] = spacing

        # Info about the overlap of the prediction with all groundtruths of sample
        if len(groundtruths) != 0:
            dices = list(dice_arr[prediction["index"]])
            gt_sizes = list(int(gt["size"]) for gt in groundtruths)
            precisions = list(prec_arr[prediction["index"]])
            recalls = list(reca_arr[prediction["index"]])
            intersects = list(inte_arr[prediction["index"]])
            unions = list(unio_arr[prediction["index"]])

            pairs = [
                (
                    float(dice),
                    int(gt_size),
                    int(n_intersect),
                    int(n_union),
                    float(prec),
                    float(rec),
                )
                for dice, gt_size, n_intersect, n_union, prec, rec in zip(
                    dices, gt_sizes, intersects, unions, precisions, recalls
                )
            ]

            # Info about the maximum matched gt
            # 0: dice, 1: gt_size, 2: n_intersect, 3: n_union, 4: precision, 5: recall
            max_dice_gt_index = int(np.argmax(dices))
            max_dice = float(dices[max_dice_gt_index])
            res_dict["dice_size_intersect_union_precision_recall_pairs"] = pairs
            res_dict["max_dice"] = max_dice
            res_dict["max_dice_gt_index"] = max_dice_gt_index
            res_dict["max_dice_gt_voxels"] = pairs[max_dice_gt_index][1]
            res_dict["max_dice_n_intersect"] = pairs[max_dice_gt_index][2]
            res_dict["max_dice_n_union"] = pairs[max_dice_gt_index][3]
            res_dict["max_dice_precision"] = pairs[max_dice_gt_index][4]
            res_dict["max_dice_recall"] = pairs[max_dice_gt_index][5]
        else:
            res_dict["dice_size_intersect_union_precision_recall_pairs"] = [(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)]
            res_dict["max_dice"] = 0.0
            res_dict["max_dice_gt_index"] = -1
            res_dict["max_dice_gt_voxels"] = np.NaN
            res_dict["max_dice_n_intersect"] = 0.0
            res_dict["max_dice_n_union"] = 0.0
            res_dict["max_dice_precision"] = 0.0
            res_dict["max_dice_recall"] = 0.0
        matched_predictions.append(res_dict)
    return matched_predictions


def match_groundtruths(
    predictions: List[Dict],
    groundtruths: List[Dict],
    spacing: np.ndarray,
    all_to_all_comps: Dict,
    sample_name: str,
) -> List[Dict]:
    matched_groundtruths = []
    vol_per_voxel = float(np.prod(spacing))
    dice_arr = all_to_all_comps["dice"]
    prec_arr = all_to_all_comps["precision"]
    reca_arr = all_to_all_comps["recall"]
    inte_arr = all_to_all_comps["intersection"]
    unio_arr = all_to_all_comps["union"]
    for groundtruth in groundtruths:
        res_dict = dict()

        res_dict["type"] = "groundtruth"
        res_dict["image_name"] = sample_name
        gt_index = groundtruth["index"]
        res_dict["groundtruth_size"] = groundtruth["size"]
        res_dict["groundtruth_index"] = gt_index
        res_dict["volume_per_voxel_mm3"] = vol_per_voxel
        res_dict["spacing"] = spacing

        if len(predictions) != 0:
            dices = list(dice_arr[:, gt_index])
            pd_sizes = [int(pd["size"]) for pd in predictions]
            precisions = list(prec_arr[:, gt_index])
            recalls = list(reca_arr[:, gt_index])
            intersects = list(inte_arr[:, gt_index])
            unions = list(unio_arr[:, gt_index])

            pairs = [
                (
                    float(dice),
                    int(pd_size),
                    int(n_intersect),
                    int(n_union),
                    float(prec),
                    float(rec),
                )
                for dice, pd_size, n_intersect, n_union, prec, rec in zip(
                    dices, pd_sizes, intersects, unions, precisions, recalls
                )
            ]

            max_dice = float(np.max(dices))
            max_dice_pd_index = int(np.argmax(dices))

            res_dict["dice_size_intersect_union_precision_recall_pairs"] = pairs
            res_dict["max_dice"] = max_dice
            res_dict["max_dice_pd_index"] = max_dice_pd_index
            res_dict["max_dice_pd_voxels"] = pairs[max_dice_pd_index][1]
            res_dict["max_dice_n_intersect"] = pairs[max_dice_pd_index][2]
            # 0: dice, 1: gt_size, 2: n_intersect, 3: n_union, 4: precision, 5: recall
            res_dict["max_dice_n_union"] = pairs[max_dice_pd_index][3]
            res_dict["max_dice_precision"] = pairs[max_dice_pd_index][4]
            res_dict["max_dice_recall"] = pairs[max_dice_pd_index][5]
        else:
            res_dict["dice_size_intersect_union_precision_recall_pairs"] = [
                (0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
            ]  # Previously dice_size_pairs
            res_dict["max_dice"] = 0.0
            res_dict["max_dice_pd_index"] = -1
            res_dict["max_dice_pd_voxels"] = np.NaN
            res_dict["max_dice_n_intersect"] = 0.0  # 0: dice, 1: gt_size, 2: n_intersect, 3: n_union,
            # 4: precision, 5: recall
            res_dict["max_dice_n_union"] = 0.0
            res_dict["max_dice_precision"] = 0.0
            res_dict["max_dice_recall"] = 0.0
        matched_groundtruths.append(res_dict)
    return matched_groundtruths


def match_strictly(
    predictions: List[Dict],
    groundtruths: List[Dict],
    dice_threshold: float,
    spacing: np.ndarray,
    all_to_all_comps: Dict,
    sample_name: str,
):
    results = []

    vol_per_voxel = float(np.prod(spacing))
    # Prediction in rows and GTs in Cols.
    dice_arr = all_to_all_comps["dice"]
    prec_arr = all_to_all_comps["precision"]
    reca_arr = all_to_all_comps["recall"]
    inte_arr = all_to_all_comps["intersection"]
    unio_arr = all_to_all_comps["union"]

    # -------------------------- Create padded arrays ------------------------- #
    if len(predictions) == 0 and len(groundtruths) == 0:
        res_dict = dict()
        res_dict["matching"] = "strict"
        res_dict["image_name"] = sample_name
        res_dict["dice"] = np.NaN
        res_dict["precision"] = np.NaN
        res_dict["recall"] = np.NaN
        res_dict["pq_weight"] = np.NaN
        res_dict["intersection_voxels"] = np.NaN
        res_dict["union_voxels"] = np.NaN
        res_dict["prediction_voxels"] = np.NaN
        res_dict["prediction_volume"] = np.NaN
        res_dict["prediction_index"] = np.NaN
        res_dict["groundtruth_voxels"] = np.NaN
        res_dict["groundtruth_volume"] = np.NaN
        res_dict["groundtruth_index"] = np.NaN
        res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
        res_dict["pred-gt_volume"] = np.NaN
        res_dict["spacing"] = spacing
        res_dict["true_positive"] = np.NaN
        res_dict["false_negative"] = np.NaN
        res_dict["match"] = np.NaN
        results.append(res_dict)
    elif len(predictions) == 0:
        for gt in groundtruths:
            res_dict = dict()
            res_dict["matching"] = "strict"
            res_dict["image_name"] = sample_name
            res_dict["dice"] = 0.0  # Weighted through panoptic quality score
            res_dict["precision"] = 0.0  # Weighted through panoptic quality score
            res_dict["recall"] = 0.0  # Weighted through panoptic quality score
            res_dict["pq_weight"] = 0.5
            res_dict["intersection_voxels"] = 0
            res_dict["union_voxels"] = gt["size"]
            res_dict["prediction_voxels"] = np.NaN
            res_dict["prediction_volume"] = np.NaN
            res_dict["prediction_index"] = np.NaN
            res_dict["groundtruth_voxels"] = gt["size"]
            res_dict["groundtruth_volume"] = float(gt["size"] * vol_per_voxel)
            res_dict["groundtruth_index"] = gt["index"]
            res_dict["pred-gt_volume"] = np.NaN
            res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
            res_dict["spacing"] = spacing
            res_dict["true_positive"] = np.NaN
            res_dict["false_negative"] = True
            res_dict["match"] = False
            results.append(res_dict)
    elif len(groundtruths) == 0:
        for pd in predictions:
            res_dict = dict()
            res_dict["matching"] = "strict"
            res_dict["image_name"] = sample_name
            res_dict["dice"] = 0.0  # Weighted through panoptic quality score
            res_dict["precision"] = 0.0  # Weighted through panoptic quality score
            res_dict["recall"] = 0.0  # Weighted through panoptic quality score
            res_dict["pq_weight"] = 0.5
            res_dict["intersection_voxels"] = 0
            res_dict["union_voxels"] = pd["size"]
            res_dict["prediction_voxels"] = pd["size"]
            res_dict["prediction_volume"] = float(pd["size"] * vol_per_voxel)
            res_dict["prediction_index"] = pd["index"]
            res_dict["groundtruth_voxels"] = np.NaN
            res_dict["groundtruth_volume"] = np.NaN
            res_dict["groundtruth_index"] = np.NaN
            res_dict["pred-gt_volume"] = np.NaN
            res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
            res_dict["spacing"] = spacing
            res_dict["true_positive"] = False
            res_dict["false_negative"] = np.NaN
            res_dict["match"] = False
            results.append(res_dict)
    else:
        row_ids, col_ids = lsa(-dice_arr)

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

            res_dict = dict()
            res_dict["matching"] = "strict"
            res_dict["image_name"] = sample_name
            res_dict["dice"] = float(dice_arr[row, col])
            res_dict["precision"] = float(prec_arr[row, col])
            res_dict["recall"] = float(reca_arr[row, col])
            res_dict["intersection_voxels"] = int(inte_arr[row, col])
            res_dict["union_voxels"] = int(unio_arr[row, col])
            res_dict["prediction_voxels"] = pred["size"]
            res_dict["prediction_volume"] = float(pred["size"] * vol_per_voxel)
            res_dict["prediction_index"] = pred["index"]
            res_dict["groundtruth_voxels"] = gt["size"]
            res_dict["groundtruth_volume"] = float(gt["size"] * vol_per_voxel)
            res_dict["pred-gt_volume"] = float((pred["size"] - gt["size"]) * vol_per_voxel)
            res_dict["groundtruth_index"] = gt["index"]
            res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
            res_dict["spacing"] = spacing
            res_dict["pq_weight"] = 1.0
            res_dict["match"] = True

            if res_dict["dice"] > dice_threshold:  # Always a TP if dice > 0.1
                res_dict["true_positive"] = True
                res_dict["false_negative"] = False
            else:
                res_dict["true_positive"] = False
                res_dict["false_negative"] = True
            results.append(res_dict)
        for nmp in not_matched_preds:
            pred = predictions[nmp]
            res_dict = dict()
            res_dict["matching"] = "strict"
            res_dict["image_name"] = sample_name
            res_dict["dice"] = 0.0
            res_dict["precision"] = 0.0
            res_dict["recall"] = 0.0
            res_dict["pq_weight"] = 0.5
            res_dict["intersection_voxels"] = 0
            res_dict["union_voxels"] = pred["size"]
            res_dict["prediction_voxels"] = pred["size"]
            res_dict["prediction_volume"] = float(pred["size"] * vol_per_voxel)
            res_dict["prediction_index"] = pred["index"]
            res_dict["groundtruth_voxels"] = np.NaN
            res_dict["groundtruth_volume"] = np.NaN
            res_dict["groundtruth_index"] = np.NaN
            res_dict["pred-gt_volume"] = np.NaN
            res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
            res_dict["spacing"] = spacing
            res_dict["true_positive"] = False
            res_dict["false_negative"] = np.NaN
            res_dict["match"] = False
            results.append(res_dict)
        for nmg in not_matched_gts:
            gt = groundtruths[nmg]
            res_dict = dict()
            res_dict["matching"] = "strict"
            res_dict["image_name"] = sample_name
            res_dict["dice"] = 0.0
            res_dict["precision"] = 0.0
            res_dict["recall"] = 0.0
            res_dict["pq_weight"] = 0.5
            res_dict["intersection_voxels"] = 0
            res_dict["union_voxels"] = gt["size"]
            res_dict["prediction_voxels"] = np.NaN
            res_dict["prediction_volume"] = np.NaN
            res_dict["prediction_index"] = np.NaN
            res_dict["groundtruth_voxels"] = gt["size"]
            res_dict["groundtruth_volume"] = float(gt["size"] * vol_per_voxel)
            res_dict["groundtruth_index"] = gt["index"]
            res_dict["pred-gt_volume"] = np.NaN
            res_dict["volume_per_voxel_mm3"] = float(vol_per_voxel)
            res_dict["spacing"] = spacing
            res_dict["true_positive"] = np.NaN
            res_dict["false_negative"] = True
            res_dict["match"] = False
            results.append(res_dict)
    return results


def calc_hard_matches(
    pd_label_flat_np, gt_label_flat_np, dice_threshold: float, spacing: np.ndarray, sample_name: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    :param pd_label_flat_np: flattend prediction instances of the network.
    :param gt_label_flat_np: flattend groundtruth instances of the network.
    :param sample_name: Name of the current sample
    :return: List of matched predictions and List of matched groundtruths
    """

    # read images, discard where both images are background (most of the image)
    # ~100x Speedup by doing this (depends on sparsity)
    non_zero = (pd_label_flat_np != 0) | (gt_label_flat_np != 0)
    # If neither gt nor pd values exist for this sample.
    if np.count_nonzero(non_zero) == 0:
        return [get_empty_strict(sample_name)]
    else:
        pd_label_flat_np = pd_label_flat_np[non_zero]
        gt_label_flat_np = gt_label_flat_np[non_zero]

        # Collect the instances with the volume and number of instance
        predictions = find_instances(pd_label_flat_np)
        groundtruths = find_instances(gt_label_flat_np)

        all_to_all_comp = compare_all_pd_to_gt(
            predictions=predictions,
            groundtruths=groundtruths,
            flat_pd_npy=pd_label_flat_np,
            flat_gt_npy=gt_label_flat_np,
        )

        one2one_matches = match_strictly(
            predictions, groundtruths, dice_threshold, spacing, all_to_all_comp, sample_name
        )
        return one2one_matches


def calc_soft_matches(
    pd_label_flat_np, gt_label_flat_np, dice_threshold, spacing, sample_name: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    :param pd_label_flat_np: flattend prediction instances of the network.
    :param gt_label_flat_np: flattend groundtruth instances of the network.
    :param sample_name: Name of the current sample
    :return: List of matched predictions and List of matched groundtruths
    """

    # read images, discard where both images are background (most of the image)
    # ~100x Speedup by doing this (depends on sparsity)
    non_zero = (pd_label_flat_np != 0) | (gt_label_flat_np != 0)
    # If neither gt nor pd values exist for this sample.
    if np.count_nonzero(non_zero) == 0:
        matched_predictions = [get_empty_prediction(sample_name)]
        matched_groundtruths = [get_empty_groundtruth(sample_name)]
        return matched_predictions, matched_groundtruths

    pd_label_flat_np = pd_label_flat_np[non_zero]
    gt_label_flat_np = gt_label_flat_np[non_zero]

    # Collect the instances with the volume and number of instance
    predictions = find_instances(pd_label_flat_np)
    groundtruths = find_instances(gt_label_flat_np)

    all_to_all_comp = compare_all_pd_to_gt(
        predictions=predictions,
        groundtruths=groundtruths,
        flat_pd_npy=pd_label_flat_np,
        flat_gt_npy=gt_label_flat_np,
    )
    matched_predictions = match_predictions(predictions, groundtruths, spacing, all_to_all_comp, sample_name)
    matched_groundtruths = match_groundtruths(predictions, groundtruths, spacing, all_to_all_comp, sample_name)
    return matched_predictions, matched_groundtruths


def calculate_tps_and_fps_v1_sparse(
    predicted_label_path: str, groundtruth_label_path: str, match_style: str = "hard", dice_threshold: float = 0.1
) -> Tuple[List[Dict], List[Dict]]:  # V1: 5.6091063022613525 seconds for loop over T67
    """Expects only predictions and groundtruth labels of actual instances,
    background = 0, other instances = {1...N}

    :param predicted_label_path:
    :param groundtruth_label_path:
    :return:
    """
    # pr = cProfile.Profile()
    # pr.enable()

    ######### Getting Paths for Labeled Prediction Sample #########
    # Loading image data
    patient_name = os.path.basename(groundtruth_label_path)

    pd_label_flat_np: np.ndarray = get_flat_array_from_image(predicted_label_path)
    pd_spacing: np.ndarray = get_spacing_from_image(predicted_label_path)
    gt_spacing: np.ndarray = get_spacing_from_image(groundtruth_label_path)
    assert np.allclose(pd_spacing, gt_spacing), "Spacing of pd and gt do not match!"
    gt_label_flat_np: np.ndarray = get_flat_array_from_image(groundtruth_label_path)
    if match_style == "hard":
        one2one_matches = calc_hard_matches(
            pd_label_flat_np, gt_label_flat_np, dice_threshold, pd_spacing, patient_name
        )
        return one2one_matches
    elif match_style == "soft":
        matched_pd, matched_gt = calc_soft_matches(
            pd_label_flat_np, gt_label_flat_np, dice_threshold, pd_spacing, patient_name
        )
        return matched_pd, matched_gt
    else:
        raise NotImplementedError("Only hard and soft matching implemented")


def run_tp_fp_calculation(
    pd_labeled: list,
    gt_labeled: list,
    dice_threshold: float,
    config_strings: str,
    n_processes: int,
    data_pair: conf.DataPair,
):
    config_instancewise_p = os.path.join(data_pair.instancewise_result_p, config_strings)
    os.makedirs(config_instancewise_p, exist_ok=True)
    for match_style in ["hard", "soft"]:
        if n_processes == 1:
            joint_result = []
            for pred, gt in zip(pd_labeled, gt_labeled):
                joint_result.append(calculate_tps_and_fps_v1_sparse(pred, gt, match_style, dice_threshold))
        else:
            with Pool(n_processes) as p:
                joint_result = p.starmap(
                    calculate_tps_and_fps_v1_sparse,
                    zip(pd_labeled, gt_labeled, repeat(match_style), repeat(dice_threshold)),
                )
        if match_style == "hard":
            joint_result = list(chain(*joint_result))
            ff.save_json(joint_result, file=os.path.join(config_instancewise_p, conf.hard_result_template))
            pd.DataFrame(joint_result).to_csv(os.path.join(config_instancewise_p, "hard_result.csv"))
        elif match_style == "soft":
            prediction_results = []
            groundtruth_results = []
            [prediction_results.extend(pd_res) for pd_res, _ in joint_result]
            [groundtruth_results.extend(gt_res) for _, gt_res in joint_result]
            ff.save_json(
                prediction_results,
                file=os.path.join(config_instancewise_p, conf.pd_result_template),
            )
            pd.DataFrame(prediction_results).to_csv(os.path.join(config_instancewise_p, "soft_result_prediction.csv"))
            ff.save_json(
                groundtruth_results,
                file=os.path.join(config_instancewise_p, conf.gt_result_template),
            )
            pd.DataFrame(prediction_results).to_csv(
                os.path.join(config_instancewise_p, "soft_result_groundtruth.csv")
            )
        else:
            raise NotImplementedError("Placeholder if made parser arg at some point.")
    return


def match_segmentations():
    sample_names = get_all_sample_names()
    sample_names.sort()
    for gt_kernel_name, _ in conf.kernel_labeling:
        for gt_dilation_size in range(8):
            for pd_dilation_size in range(8):
                for pd_kernel_name, _ in conf.kernel_labeling:
                    groundtruth_dir = os.path.join(
                        conf.train_label_gt_location,
                        conf.param_name_convention.format(gt_kernel_name, gt_dilation_size),
                    )
                    source_gt = [os.path.join(groundtruth_dir, sample) for sample in sample_names]

                    print(
                        "Evaluating Accuracy: gt_Kernel: {}, gt_n_dilation {}, pd_Kernel: {}, pd_n_dilation: {},".format(
                            gt_kernel_name,
                            gt_dilation_size,
                            gt_kernel_name,
                            pd_dilation_size,
                        )
                    )
                    prediction_dir = os.path.join(
                        conf.train_label_pd_location,
                        conf.param_name_convention.format(pd_kernel_name, pd_dilation_size),
                    )
                    source_pd = [os.path.join(prediction_dir, sample) for sample in sample_names]

                    result_dir = os.path.join(
                        conf.result_location,
                        conf.prediction_groundtruth_matching_template.format(
                            gt_kernel_name,
                            gt_dilation_size,
                            pd_kernel_name,
                            pd_dilation_size,
                        ),
                    )
                    os.makedirs(result_dir, exist_ok=True)

                    prediction_result_path = os.path.join(result_dir, conf.pd_result_template)
                    groundtruth_result_path = os.path.join(result_dir, conf.gt_result_template)

                    if os.path.exists(prediction_result_path) and os.path.exists(groundtruth_result_path):
                        continue

                    p = Pool(24)
                    starttime_2 = time()
                    joint_result = p.starmap(calculate_tps_and_fps_v1_sparse, zip(source_pd, source_gt))

                    p.close()
                    p.join()
                    endtime_2 = time()
                    prediction_res = []
                    groundtruth_res = []
                    [prediction_res.extend(pd_res) for pd_res, gt_res in joint_result]
                    [groundtruth_res.extend(gt_res) for pd_res, gt_res in joint_result]

                    print("V1: {} seconds".format(endtime_2 - starttime_2))

                    ff.save_json(prediction_res, prediction_result_path)
                    ff.save_json(groundtruth_res, groundtruth_result_path)
    return


if __name__ == "__main__":
    match_segmentations()
