import os
from pathlib import Path
from batchgenerators.utilities import file_and_folder_operations as ff
import numpy as np
import pandas as pd

from nneval.utils.io import write_dict_to_json


def load_hard_match(configuration: str, result_path: str):
    current_conf_path = os.path.join(result_path, configuration)
    hard_results = ff.load_json(os.path.join(current_conf_path, conf.hard_result_template))
    return hard_results


def lesion_wise_eval(pd_df: pd.DataFrame):
    unique_image = pd_df["image_name"].unique()
    all_img_results = []
    for im_name in unique_image:
        tmp_df = pd_df[pd_df["image_name"] == im_name]

        instance_wise_dice = np.mean(tmp_df["dice"] * tmp_df["pq_weight"])
        pixel_wise_precision = np.mean(tmp_df["precision"] * tmp_df["pq_weight"])
        pixel_wise_recall = np.mean(tmp_df["recall"] * tmp_df["pq_weight"])

        true_positives = np.nansum(tmp_df["true_positive"])
        false_positives = np.nansum(~tmp_df["true_positive"])
        n_positives = true_positives + false_positives
        case_precision = true_positives / n_positives

        all_gt_instances = np.nansum(tmp_df["true_positive"]) + np.nansum(tmp_df["false_negative"])

        case_recall = true_positives / all_gt_instances

        img_result = {
            "image_name": im_name,
            "instance_wise_dice": instance_wise_dice,
            "pixel_wise_precision": pixel_wise_precision,
            "pixel_wise_recall": pixel_wise_recall,
            "image_precision": case_precision,
            "image_recall": case_recall,
            "image_f1": 2 * (case_precision * case_recall) / (case_precision + case_recall),
        }
        all_img_results.append(img_result)

    return all_img_results


def lesion_wise_case_wise_averaging(df: pd.DataFrame):
    # This is also debateable: Not matched instances & groundtruths count half with their zeros. 
    #   This is since they have no partner, a lesion that has a really low overlap would be worse
    #   Otherwise matching two instances with 0 would be better than not "matched" instances
    #   Properly dealing with this is not trivial, unfortunately
    lw_prec = (df["precision"] * df["pq_weight"]).sum() / (df["pq_weight"]).sum()
    lw_rec = (df["recall"] * df["pq_weight"]).sum() / (df["pq_weight"]).sum()
    lw_dice = (df["dice"] * df["pq_weight"]).sum() / (df["pq_weight"]).sum()
    if (np.isnan(lw_prec) or np.isnan(lw_rec) or np.isnan(lw_dice)):
        print("Wait")
    tps = np.nansum(df["true_positive"])
    fns = np.nansum(df["false_negative"])
    fps = np.nansum(1 - df["true_positive"])

    n_gts = (~pd.isna(df["false_negative"])).sum()
    n_pds = (~pd.isna(df["true_positive"])).sum()

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
    return {
        "matching": df["matching"].iloc[0],
        "volume_per_voxel_mm3": df["volume_per_voxel_mm3"].iloc[0],
        "lw_precision": lw_prec,
        "lw_recall": lw_rec,
        "lw_dice": lw_dice,
        "case_precision": case_precision,
        "case_recall": case_recall,
        "case_f1": case_f1,
    }


def calculate_statistics(
    pd_df: pd.DataFrame,
    column_keys: list[str] = ["lw_precision", "lw_recall", "lw_dice", "case_precision", "case_recall", "case_f1"],
) -> dict:
    """Calculate the statistics for the dataframe"""
    full_json = {}
    for ck in column_keys:
        res_json = {}
        res_json["mean"] = float(pd_df[ck].mean())
        res_json["std"] = float(pd_df[ck].std())
        res_json["min"] = float(pd_df[ck].min())
        res_json["max"] = float(pd_df[ck].max())

        res_json["percentile_10"] = float(pd_df[ck].quantile(0.05))
        res_json["percentile_25"] = float(pd_df[ck].quantile(0.25))
        res_json["percentile_50"] = float(pd_df[ck].quantile(0.5))
        res_json["percentile_75"] = float(pd_df[ck].quantile(0.75))
        res_json["percentile_90"] = float(pd_df[ck].quantile(0.95))
        full_json[ck] = res_json
    return full_json


def calculate_gt_statistics(
    pd_df: pd.DataFrame,
) -> dict:
    """Calculate the statistics for the dataframe"""
    vol_info = calculate_statistics(pd_df, column_keys=["groundtruth_volume"])
    case_count_info = pd_df.groupby("image_name").apply(lambda x: max(x["groundtruth_index"]) + 1).values
    # case_count_info = pd_df.groupby("image_name").apply(lambda x: sum(0 if pd.isna(x["false_negative"]) else 1 ))
    case_count_df = pd.DataFrame()
    case_count_df["case_count"] = case_count_info
    case_count_info = calculate_statistics(case_count_df, column_keys=["case_count"])
    groundtruth_info = {"groundtruth_instance_volume": vol_info, "groundtruth_case_count": case_count_info}
    return groundtruth_info


def create_patient_wise_values(pd_df: pd.DataFrame):
    """Takes the sum of lesion-wise (pixel) precision, lesion-wise (pixel) recall, lesion-wise dice  and divide it by the sum of the pq weight, for each image_name in the dataframe"""
    tmp_df = pd_df.drop(columns=["spacing"])
    results_dicts = tmp_df.groupby("image_name").apply(lesion_wise_case_wise_averaging)
    result_df = pd.DataFrame(results_dicts.to_list(), index=results_dicts.index)
    result_df.reset_index(inplace=True)
    lswcw_stats = calculate_statistics(result_df)

    return result_df, lswcw_stats


def lesion_wise_dataset_wide_eval(pd_df):
    panoptic_dice = float(np.nansum(pd_df["dice"] * pd_df["pq_weight"]) / np.nansum(pd_df["pq_weight"]))
    true_positives = int(np.nansum(pd_df["true_positive"]))
    false_positives = int(np.nansum(1 - pd_df["true_positive"]))
    false_negatives = int(np.nansum(pd_df["false_negative"]))
    total_gt_instances = int(np.nansum(1 - pd_df["false_negative"]) + np.nansum(pd_df["false_negative"]))
    assert total_gt_instances == (true_positives + false_negatives), "GTs and TPs + FNs not equal"
    n_preds = int(true_positives + false_positives)
    dataset_precision = float(true_positives / n_preds)
    dataset_recall = true_positives / total_gt_instances
    dataset_f1 = 2 * (dataset_precision * dataset_recall) / (dataset_precision + dataset_recall)
    gt_statistics = calculate_gt_statistics(pd_df)

    matched_df = pd_df[pd_df["match"]]
    dataset_stats = calculate_statistics(
        matched_df, column_keys=["dice", "precision", "recall", "prediction_volume", "groundtruth_volume"]
    )
    result = {
        "panoptic_dice": panoptic_dice,
        "dataset_precision": dataset_precision,
        "dataset_recall": dataset_recall,
        "dataset_f1": dataset_f1,
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "n_prediction": n_preds,
        "n_groundtruths": total_gt_instances,
        "statistics_true_positives": dataset_stats,
        **gt_statistics,
    }
    return result


def hard_evaluate(
    configuration: str,
    result_path: str,
    samples: list[str],
    filtering_size: int | None = None,
):
    """ """
    # ----------------------------------- Path ----------------------------------- #
    current_conf_path = os.path.join(result_path, configuration, "hard_eval")
    Path(current_conf_path).mkdir(exist_ok=True, parents=True)
    # --------------------------------- DataFrame -------------------------------- #
    hard_results = load_hard_match(configuration, result_path)
    pd_df = pd.DataFrame(hard_results)

    # ----------------------- Lesion wise full dataset wide ---------------------- #
    ds_json = lesion_wise_dataset_wide_eval(pd_df)
    write_dict_to_json(ds_json, os.path.join(current_conf_path, "dataset_wide_values.json"))
    # 1. Create histogram of dice over volume
    # 2. Create csv of instance-wise patient-wise values (Dice, F1, Precision, Recall)

    # ------------------------- Patient wise dataset wise ------------------------ #
    pwdw_df, pwdw_json = create_patient_wise_values(pd_df)
    pwdw_df.to_csv(os.path.join(current_conf_path, "lw_cw_values.csv"))
    write_dict_to_json(pwdw_json, os.path.join(current_conf_path, "lw_cw_values_stats.json"))

    return
