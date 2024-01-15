from __future__ import annotations

import os
from pathlib import Path
import re
from deprecated import deprecated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batchgenerators.utilities import file_and_folder_operations as ff

import nneval.utils.configuration as conf
from multiprocessing import Pool
from nneval.deprecated.evaluation import plot_true_false_histograms
from nneval.deprecated.evaluation import csv_analysis, plot_size_dice_correlation
from nneval.deprecated.evaluation.csv_analysis import case_wise_summary_json
from nneval.deprecated.evaluation.plot_true_false_histograms import create_gt_volume_over_dice, create_pd_volume_over_dice
from nneval.deprecated.utilities import save_list_dict_to_csv


def load_predictions_and_groundtruths(configuration: str, result_path: str):
    current_conf_path = os.path.join(result_path, configuration)
    prediction_results = ff.load_json(os.path.join(current_conf_path, conf.pd_result_template))
    groundtruth_results = ff.load_json(os.path.join(current_conf_path, conf.gt_result_template))
    return prediction_results, groundtruth_results


@deprecated
def soft_evaluate(
    configuration: str,
    result_path: str,
    samples: list[str],
    dice_threshold: float = 0.1,
    filtering_size: int | None = None,
):
    """ """

    prediction_results, groundtruth_results = load_predictions_and_groundtruths(configuration, result_path)
    current_conf_path = os.path.join(result_path, configuration, "soft_eval")
    Path(current_conf_path).mkdir(exist_ok=True, parents=True)

    pd_df = pd.DataFrame(prediction_results)
    gt_df = pd.DataFrame(groundtruth_results)

    create_pd_volume_over_dice(pd_df, dice_threshold=dice_threshold, save_directory=current_conf_path)
    create_gt_volume_over_dice(gt_df, dice_threshold=dice_threshold, save_directory=current_conf_path)

    plot_true_false_histograms.create_size_histrogram(pd_df, gt_df, dice_threshold, save_directory=current_conf_path)
    plot_true_false_histograms.create_size_histrogram(pd_df, gt_df, 0.25, save_directory=current_conf_path)

    plot_size_dice_correlation.create_size_dice_scatter_plot(pd_df=pd_df, gt_df=gt_df)

    # Evaluation
    size_wise_matching = csv_analysis.size_wise_matching(
        groundtruth_results=groundtruth_results,
        prediction_results=prediction_results,
        samples=samples,
    )
    average_dice_per_pd_size = csv_analysis.average_dice_per_prediction_size(prediction_results=prediction_results)
    average_dice_per_gt_size = csv_analysis.average_dice_per_groundtruth_size(groundtruth_results=groundtruth_results)
    n_gt_metastases_per_patient = csv_analysis.groundtruth_metastases_per_patient(groundtruth_results, samples)
    n_pd_metastases_per_patient = csv_analysis.groundtruth_metastases_per_patient(prediction_results, samples)
    groundtruth_volume_prediction_volume_pairs = csv_analysis.groundtruth_volume_prediction_volume_pairs(
        groundtruth_results=groundtruth_results, samples=samples
    )

    case_wise_stats_filter_size_sweep = []
    for i in range(200):
        res = csv_analysis.casewise_results(
            groundtruth_results=groundtruth_results,
            prediction_results=prediction_results,
            samples=samples,
            filter_size=i,
            dice_threshold=dice_threshold,
        )
        case_wise_stats_filter_size_sweep.append(csv_analysis.casewise_stats(res, filter_size=i))

    f1_scores = np.array([entry["mean_f1_score"] for entry in case_wise_stats_filter_size_sweep])
    if filtering_size is None:
        filter_size = int(np.argmax(f1_scores))  # Lowest threshold giving best results
    else:
        filter_size = filtering_size

    case_wise_tp_fp_fns_no_filter = csv_analysis.casewise_results(
        groundtruth_results=groundtruth_results,
        prediction_results=prediction_results,
        samples=samples,
        filter_size=0,
        dice_threshold=dice_threshold,
    )

    case_wise_no_filter_summary = case_wise_summary_json(case_wise_tp_fp_fns_no_filter)
    ff.save_json(
        case_wise_no_filter_summary,
        os.path.join(current_conf_path, "case_wise_no_filter_summary.json"),
        sort_keys=False,
    )

    case_wise_tp_fp_fns_filter = csv_analysis.casewise_results(
        groundtruth_results=groundtruth_results,
        prediction_results=prediction_results,
        samples=samples,
        filter_size=filter_size,
        dice_threshold=dice_threshold,
    )

    tmp = csv_analysis.casewise_stats(case_wise_tp_fp_fns_filter, filter_size=6)

    volume_pairs = csv_analysis.groundtruth_volume_prediction_volume_pairs(
        groundtruth_results, samples, size_filter_val=filter_size
    )

    save_list_dict_to_csv(
        volume_pairs,
        output_path=current_conf_path,
        filename="groundtruth_volume_prediction_volume_pairs_size_threshold_{}.csv".format(filtering_size),
    )

    save_list_dict_to_csv(
        prediction_results,
        output_path=current_conf_path,
        filename="all_prediction_instances.csv",
    )
    save_list_dict_to_csv(
        groundtruth_results,
        output_path=current_conf_path,
        filename="all_groundtruth_instances.csv",
    )
    save_list_dict_to_csv(
        size_wise_matching,
        output_path=current_conf_path,
        filename="size_wise_matching.csv",
    )
    save_list_dict_to_csv(
        average_dice_per_pd_size,
        output_path=current_conf_path,
        filename="average_dice_per_pd_size.csv",
    )
    save_list_dict_to_csv(
        average_dice_per_gt_size,
        output_path=current_conf_path,
        filename="average_dice_per_gt_size.csv",
    )
    save_list_dict_to_csv(
        groundtruth_volume_prediction_volume_pairs,
        output_path=current_conf_path,
        filename="groundtruth_volume_prediction_volume_pairs_size_threshold_0.csv",
    )
    save_list_dict_to_csv(
        n_gt_metastases_per_patient,
        output_path=current_conf_path,
        filename="gt_number_ce_per_patient.csv",
    )
    save_list_dict_to_csv(
        n_pd_metastases_per_patient,
        output_path=current_conf_path,
        filename="pd_number_ce_per_patient.csv",
    )
    save_list_dict_to_csv(
        case_wise_tp_fp_fns_filter,
        output_path=current_conf_path,
        filename="case_wise_tp_fp_fn.csv",
    )
    save_list_dict_to_csv(
        case_wise_tp_fp_fns_no_filter,
        output_path=current_conf_path,
        filename="case_wise_tp_fp_fn_no_filter.csv",
    )
    save_list_dict_to_csv(
        case_wise_stats_filter_size_sweep,
        output_path=current_conf_path,
        filename="case_wise_stats.csv",
    )

    f1_scores = []
    filter_size = []
    for result in size_wise_matching:
        f1_scores.append(result["f1_score"])
        filter_size.append(result["prediction_filter_size"])

    max_f1_score_id = int(np.argmax(f1_scores))
    max_f1_score_filter_size = filter_size[max_f1_score_id]
    max_f1_score = f1_scores[max_f1_score_id]

    #  pd_sizes, gt_sizes, match_dice = plot_match_pd_gt_size_scatter.match_pd_gt_sizes_dice_cube_root(pd_df, gt_df, dice_threshold=0.)

    return max_f1_score_filter_size, max_f1_score, configuration


def main(configurations: list[str]):
    result_location = conf.result_location
    if len(configurations) == 0:
        configurations = sorted(os.listdir(result_location))
        result_configurations = [
            os.path.join(result_location, entry)
            for entry in configurations
            if re.match(
                "gtKernel_(rectangle|ball|cross)_Ndilation_\d+_pdKernel_(rectangle|ball|cross)_Ndilation_\d+",
                entry,
            )
        ]
    else:
        result_configurations = [os.path.join(result_location, conf) for conf in configurations]
    dice_threshold = 0.1
    max_filter_size = 200

    p = Pool(24)
    all_results = p.starmap(
        soft_evaluate,
        zip(
            result_configurations,
            [result_location for _ in result_configurations],
            [dice_threshold for _ in result_configurations],
            [max_filter_size for _ in result_configurations],
        ),
    )

    all_results_descending_by_dice = sorted(all_results, key=lambda content: content[1])[::-1]

    all_dict_results = []
    for result in all_results_descending_by_dice:
        all_dict_results.append(
            dict(
                prediction_size_filter=result[0],
                f1_score=result[1],
                configuration=result[2],
            )
        )
    ff.save_json(all_dict_results, os.path.join(result_location, "configwise_f1_scores.json"))

    max_scores = [max_f1_score for filter_size, max_f1_score, config in all_results]
    max_f1_filter_size = [filter_size for filter_size, max_f1_score, config in all_results]

    plt.scatter(x=max_f1_filter_size, y=max_scores)
    plt.savefig(os.path.join(result_location, "scatter_f1_scores.svg"))
    plt.close()


if __name__ == "__main__":
    optimal_configuration = conf.prediction_groundtruth_matching_template.format("ball", 3, "ball", 3)
    main(configurations=[optimal_configuration])
