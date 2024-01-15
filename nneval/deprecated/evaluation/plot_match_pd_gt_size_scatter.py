from typing import Tuple, List

import numpy as np
import pandas as pd


def match_pd_gt_sizes_dice(
    pd_df: pd.DataFrame, gt_df: pd.DataFrame, dice_threshold: float = 0.1
) -> Tuple[List[int], List[int], List[float]]:
    filtered_pd_df = pd_df[(pd_df["max_dice"] >= dice_threshold)]
    prediction_sizes = filtered_pd_df["prediction_size"].to_list()
    dice_size_pairs = filtered_pd_df[
        "'dice_size_intersect_union_precision_recall_pairs'"
    ].to_list()
    pd_size, gt_size, dice = [], [], []

    for counter, (pred_size, predictions) in enumerate(
        zip(prediction_sizes, dice_size_pairs)
    ):
        dice_vals = [dice for dice, _, _, _, _, _ in predictions]
        max_dice = np.max(dice_vals)
        if max_dice == 0:
            continue
        id = np.argmax(dice_vals)
        pd_size.append(pred_size)
        gt_size.append(predictions[id][1])
        dice.append(predictions[id][0])

    return pd_size, gt_size, dice


def match_pd_gt_sizes_dice_cube_root(
    pd_df: pd.DataFrame, gt_df: pd.DataFrame, dice_threshold: float = 0.1
) -> Tuple[List[int], List[int], List[float]]:
    filtered_pd_df = pd_df[(pd_df["max_dice"] >= dice_threshold)]
    prediction_sizes = filtered_pd_df["prediction_size"].to_list()
    dice_size_pairs = filtered_pd_df[
        "'dice_size_intersect_union_precision_recall_pairs'"
    ].to_list()
    pd_size, gt_size, dice = [], [], []

    for counter, (pred_size, predictions) in enumerate(
        zip(prediction_sizes, dice_size_pairs)
    ):
        dice_vals = [dice for dice, _, _, _, _, _ in predictions]
        max_dice = np.max(dice_vals)
        if max_dice == 0:
            continue
        id = np.argmax(dice_vals)
        pd_size.append(pred_size ** (1 / 3))
        gt_size.append(predictions[id][1] ** (1 / 3))
        dice.append(predictions[id][0])

    return pd_size, gt_size, dice
