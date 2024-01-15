import os
from warnings import warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy.stats import pearsonr


def differentiate_match_lower_dice_no_overlap(dice_series: pd.Series, dice_threshold) -> np.ndarray:
    matches = np.where(dice_series >= dice_threshold)
    low_dice_matches = np.where((dice_series < dice_threshold) & (dice_series != 0))

    match_values = np.full_like(dice_series.to_numpy(), fill_value="no match", dtype=object)
    match_values[matches] = "match"
    match_values[low_dice_matches] = "low dice"

    return match_values


def create_pd_volume_over_dice(pd_df: pd.DataFrame, dice_threshold: float, save_directory: str):
    tmp_df = pd_df.copy()
    try:
        tmp_df["log(volume)"] = np.log(tmp_df["max_dice_gt_voxels"])
    except KeyError as e:
        warn(
            "Incomplete pd_df! This can happen when no pd is matched to a gt!"
            "Not creating prediction volume over dice plot!"
        )
        print(e)
        return

    tmp_df["dice"] = tmp_df["max_dice"]

    # plain_corr_coeff = pearsonr(x=tmp_df["max_dice_gt_voxels"].to_numpy(), y=tmp_df["dice"].to_numpy())
    # log_corr_coeff = pearsonr(x=tmp_df["log(volume)"].to_numpy(), y=tmp_df["dice"].to_numpy())

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(16, 8))
    color = "#619CFF"
    ax: plt.Axes = sns.regplot(data=tmp_df, x="log(volume)", y="dice", fit_reg=True, color=color, ax=ax)
    ax.grid(True, color="w")
    ax.set_axisbelow(True)
    ax.set(facecolor="#eaeaea")
    ax.set_title("Relationship of dice to log of lesion volume ")

    ax.set_ylim(-0.05, 1.05)
    x_lim = ax.get_xlim()
    ax.plot(x_lim, [dice_threshold, dice_threshold], color="#F8766D")
    ax.set_xlim(*x_lim)

    plt.savefig(os.path.join(save_directory, "size_to_dice.png"), dpi=600)
    plt.savefig(os.path.join(save_directory, "size_to_dice.eps"), dpi=600)
    plt.close()


def create_gt_volume_over_dice(gt_df: pd.DataFrame, dice_threshold: float, save_directory: str):
    tmp_df = gt_df.copy()
    try:
        tmp_df["log(volume)"] = np.log(tmp_df["groundtruth_size"])
    except KeyError as e:
        warn("Incomplete gt_df! Not creating gt volume over dice plot!")
        print(e)
        return
    tmp_df["dice"] = tmp_df["max_dice"]

    # plain_corr_coeff = pearsonr(x=tmp_df["groundtruth_size"].to_numpy(), y=tmp_df["dice"].to_numpy())
    # log_corr_coeff = pearsonr(x=tmp_df["log(volume)"].to_numpy(), y=tmp_df["dice"].to_numpy())

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(16, 8))
    color = "#619CFF"
    ax: plt.Axes = sns.regplot(data=tmp_df, x="log(volume)", y="dice", fit_reg=True, color=color, ax=ax)
    ax.grid(True, color="w")
    ax.set_axisbelow(True)
    ax.set(facecolor="#eaeaea")
    ax.set_title("Relationship of dice to log of lesion volume ")

    ax.set_ylim(-0.05, 1.05)
    x_lim = ax.get_xlim()
    ax.plot(x_lim, [dice_threshold, dice_threshold], color="#F8766D")
    ax.set_xlim(*x_lim)

    plt.savefig(os.path.join(save_directory, "size_to_dice.png"), dpi=600)
    plt.savefig(os.path.join(save_directory, "size_to_dice.eps"), dpi=600)
    plt.close()


def create_size_histrogram(pd_df: pd.DataFrame, gt_df: pd.DataFrame, dice_threshold: float, save_directory: str):
    """Saves an histogram showing the average cube edge length of the prediction and groundtruth and the corresponding type of match.

    :param pd_df:
    :param gt_df:
    :param dice_threshold:
    :param save_directory:
    :return:
    """
    pd_df["match_type"] = differentiate_match_lower_dice_no_overlap(pd_df["max_dice"], dice_threshold)
    pd_df["cube_size"] = pd_df["prediction_size"] ** (1 / 3)

    gt_df["match_type"] = differentiate_match_lower_dice_no_overlap(gt_df["max_dice"], dice_threshold)
    gt_df["cube_size"] = gt_df["groundtruth_size"] ** (1 / 3)
    
    bins = list(np.arange(1, 40, 0.25))
    # Prediction Histogram
    sns.set_palette(
        sns.color_palette(
            [
                "black",
                "firebrick",
                "darkgreen",
            ],
            n_colors=3,
        )
    )
    if len(pd_df) > 2:
        ax: plt.Axes = sns.histplot(
            data=pd_df.sort_values("max_dice"),
            x="cube_size",
            stat="count",
            kde=False,
            bins=bins,
            element="bars",
            legend=True,
            multiple="stack",
            hue="match_type",
        )
        ax.set_title("Prediction matches by prediction cube edge length - Dice Threshold: {}".format(dice_threshold))
        ax.set_xlabel("Prediction instance cube edge length")
        ax.set_ylabel("Count")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        lgnd = ax.get_legend()
        lgnd.set_title("Matched a groundtruth")
        plt.savefig(os.path.join(save_directory, "prediction_histogram_dice_{}.svg".format(dice_threshold)))
        plt.close()
    else:
        warn(f"Prediction histogram not created because DataFrame too small! Length: {len(pd_df)}")

    # Groundtruth Histogram
    if len(gt_df) > 2:
        ax: plt.Axes = sns.histplot(
            data=gt_df.sort_values("max_dice"),
            x="cube_size",
            stat="count",
            kde=False,
            bins=bins,
            element="bars",
            legend=True,
            multiple="stack",
            hue="match_type",
        )
        ax.set_title("Groundtruth matches by groundtruth cube edge length - Dice Threshold: {}".format(dice_threshold))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Groundtruth Instance Cube Size")
        ax.set_ylabel("Count")
        lgnd = ax.get_legend()
        ax.set_title("Matched type of the groundtruth")
        plt.savefig(os.path.join(save_directory, "groundtruth_histogram_dice_{}.svg".format(dice_threshold)))
        plt.close()
    else:
        warn(f"Groundtruth histogram not created because DataFrame too small! Length: {len(pd_df)}")
    return
