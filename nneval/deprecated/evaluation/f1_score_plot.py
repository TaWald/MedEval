import os

import numpy as np
from matplotlib import pyplot as plt


def plot_f1_score(
    output_path: str,
    prediction_df,
    groundtruth_df,
    dice_threshold=0.1,
    maximum_size_filter: int = 200,
):
    pd_df = prediction_df
    gt_df = groundtruth_df

    f1_scores = []
    missed_groundtruths = []
    size_filters = list(range(maximum_size_filter))

    for i in missed_groundtruths:
        filtered_pd_df = pd_df[pd_df["prediction_size"] > i]
        filtered_predictions = len(filtered_pd_df)
        prediction_tp_or_fp = (
            filtered_pd_df["max_dice"] >= dice_threshold
        ).value_counts()
        true_pd = prediction_tp_or_fp.loc[True]
        false_pd = prediction_tp_or_fp.loc[False]

        # Here I need to find out which predictions remain.

        dice_size_pairs = gt_df[
            "dice_size_intersect_union_precision_recall_pairs"
        ].to_list()
        matched_gts = np.zeros(len(dice_size_pairs), dtype=bool)
        for counter, groundtruths in enumerate(dice_size_pairs):
            for dice_val, prediction_size in groundtruths:
                if (dice_val >= dice_threshold) and (prediction_size >= i):
                    matched_gts[counter] = True
                    break

        true_gt = np.count_nonzero(matched_gts)
        false_gt = len(gt_df) - true_gt

        true_positives = true_pd
        false_positives = false_pd
        false_negatives = false_gt
        # true_negatives do not exist in a segmentation setting.

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(gt_df)
        f1_scores.append(2 * (precision * recall) / (precision + recall))
        missed_groundtruths.append(false_negatives)

    plt.plot(size_filters, f1_scores)
    plt.savefig(os.path.join(output_path, "size_base_f1score.svg"))
    plt.close()
