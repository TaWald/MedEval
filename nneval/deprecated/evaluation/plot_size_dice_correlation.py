import os
from warnings import warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


def create_size_dice_scatter_plot(pd_df: pd.DataFrame, gt_df: pd.DataFrame):
    pd_df["cube_size"] = pd_df["prediction_size"] ** (1 / 3)
    # pd_df["log_10_size"] = np.log10(pd_df["prediction_size"])
    # pd_df["log_e_size"] = np.log(pd_df["prediction_size"])
    gt_df["cube_size"] = gt_df["groundtruth_size"] ** (1 / 3)
    
    if len(pd_df) < 2:
        warn("Length of data_frame too small! Skipping Size over DICE scatter plot creation")
        return
    ax: plt.Axes = sns.scatterplot(data=pd_df, x="prediction_size", y="max_dice")
    ax.set_xscale("log")
    # ax.plot([0., pd_max_size], [0., 1.0], color="red")  # Plot diagonal.
    ax.set_title("Dice relation to prediction size")
    ax.set_xlabel("Log naturalis of the prediction voxel size.")
    ax.set_ylabel("Maximum dice of the prediction with a groundtruth.")

    # plt.show()
    plt.close()

    return
