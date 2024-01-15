from array import array

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def main():
    
    # instance_src_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/11_02_2022_EVAL/ce_evaluation/instance_evaluation/matching_result")
    # case_src_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/11_02_2022_EVAL/ce_evaluation/samplewise_evaluation")
    
    instance_src_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/11_02_2022_EVAL/ce_evaluation/instance_evaluation/matching_result")
    case_src_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/11_02_2022_EVAL/ce_evaluation/samplewise_evaluation")
    out_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/05_04_2022_EVAL/visualisierungen")
    instance_p1 = ["train_results", "test_results", "independent_test_results"]
    instance_p2 = "gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3"
    instance_file = "case_wise_tp_fp_fn_no_filter.csv"
    
    # Read Instance DataFrames (from csv)
    file_paths = [instance_src_path / p / instance_p2 / instance_file for p in instance_p1]
    train_pd = pd.read_csv(file_paths[0])
    test_pd = pd.read_csv(file_paths[1])
    indtest_pd = pd.read_csv(file_paths[2])

    # Read Casewise DataFrames (from csv)
    case_p1 = ["train_samplewise_evaluation", "test_samplewise_evaluation", "independent_test_samplewise_evaluation"]
    case_file = "samplewise_dices.csv"
    file_paths = [case_src_path / p / case_file for p in case_p1]
    case_train_pd = pd.read_csv(file_paths[0])
    case_test_pd = pd.read_csv(file_paths[1])
    case_indtest_pd = pd.read_csv(file_paths[2])

    x_name = "Datasets"
    
    train_pd[x_name] =  "Institutional\n Train Set"
    test_pd[x_name] = "Institutional\n Test Set"
    indtest_pd[x_name] = "External\n Test Set"

    case_train_pd[x_name] =  "Institutional\n Train Set"
    case_test_pd[x_name] = "Institutional\n Test Set"
    case_indtest_pd[x_name] = "External\n Test Set"
    
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12}
    )
    
    marker_size = 4
    box_width=0.05
    bw = 0.25
    my_pal = ["#00BA38",  "#F8766D", "#619CFF"]
    
    dark_pal = {"TR": "darkgreen", "TS": "darkred", "ITS": "darkblue"}
    ax: array
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    instance_df = pd.concat([train_pd, test_pd, indtest_pd])
    case_df = pd.concat([case_train_pd, case_test_pd, case_indtest_pd])
    
    y_dice_axis_name = "DICE coefficient"
    y_sensitivity_axis_name ="Sensitivity"
    instance_df[y_dice_axis_name] = instance_df["mean_pd_dice"]
    instance_df[y_sensitivity_axis_name] = instance_df["mean_pd_recall"]
    case_df[y_dice_axis_name] = case_df["dice"]
    case_df[y_sensitivity_axis_name] = case_df["sensitivity"]
    
    sns.set(rc={'axes.facecolor': "#eaeaea"})

    # Plot top left (Lesion Dice)
    axs[0,0].grid(True, color="w")
    axs[0,0].set_axisbelow(True)
    axs[0,0].set(facecolor="#eaeaea")
    sns.violinplot(data=instance_df, inner=None, x=x_name, y=y_dice_axis_name, palette=my_pal, bw=bw,
                   cut=0., zorder=.5, ax=axs[0, 0])
    sns.boxplot(
        x=x_name, y=y_dice_axis_name, data=instance_df,
        width=box_width,
        color="k",
        showcaps=False, boxprops={'facecolor': "w", "zorder": 10},
        showfliers=True, whiskerprops={'color': "k",'linewidth': 1.5, "zorder": 10}, ax=axs[0,0],
        zorder=10
        )
    axs[0,0].set_xlabel("")
    axs[0,0].set_title("Lesion-wise DICE")
    
    
    # Plot top right (Lesion Sensitivity)
    axs[0,1].grid(True, color="w")
    axs[0,1].set_axisbelow(True)
    axs[0, 1].set(facecolor="#eaeaea")
    sns.violinplot(data=instance_df, inner=None, x=x_name, y=y_sensitivity_axis_name, palette=my_pal, bw=bw,
                   cut=0., zorder=.5, ax=axs[0, 1])
    sns.boxplot(
        x=x_name, y=y_sensitivity_axis_name, data=instance_df,
        width=box_width,
        color="k",
        showcaps=False, boxprops={'facecolor': "w", "zorder": 10},
        showfliers=True, whiskerprops={'color': "k",'linewidth': 1.5, "zorder": 10}, ax=axs[0,1],
        zorder=10
        )
    axs[0,1].set_xlabel("")
    axs[0,1].set_title("Lesion-wise Sensitivity")
    
    
    # Plot bot left (Lesion Dice)
    axs[1,0].grid(True, color="w")
    axs[1, 0].set_axisbelow(True)
    axs[1, 0].set(facecolor="#eaeaea")
    sns.violinplot(
        data=case_df, inner=None, x=x_name, y=y_dice_axis_name, palette=my_pal,
        bw=bw,
        cut=0., zorder=.5, ax=axs[1, 0]
        )
    sns.boxplot(
        x=x_name, y=y_dice_axis_name, data=case_df,
        width=box_width,
        color="k",
        showcaps=False, boxprops={'facecolor': "w", "zorder": 10},
        showfliers=True, whiskerprops={'color': "k", 'linewidth': 1.5, "zorder": 10},
        ax=axs[1, 0],
        zorder=10
        )
    axs[1, 0].set_xlabel("")
    axs[1, 0].set_title("Case-wise DICE")

    # Plot top right (Lesion Sensitivity)
    axs[1,1].grid(True, color="w", zorder=.25)
    axs[1, 1].set_axisbelow(True)
    axs[1, 1].set(facecolor="#eaeaea")
    sns.violinplot(
        data=case_df, inner=None, x=x_name, y=y_sensitivity_axis_name,
        palette=my_pal, bw=bw,
        cut=0., zorder=.5, ax=axs[1, 1]
        )
    sns.boxplot(
        x=x_name, y=y_sensitivity_axis_name, data=case_df,
        width=box_width,
        color="k",
        showcaps=False, boxprops={'facecolor': "w", "zorder": 10},
        showfliers=True, whiskerprops={'color': "k", 'linewidth': 1.5, "zorder": 10},
        ax=axs[1, 1],
        zorder=10
        )
    axs[1, 1].set_xlabel("")
    axs[1, 1].set_title("Case-wise Sensitivity")
    plt.savefig(str(out_path / "joint_plot_300dpi.tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path /"joint_plot_600dpi.tiff"), dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "joint_plot_1200dpi.tiff"), dpi=1200, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path/"joint_plot_1200dpi.eps"), dpi=1200)
    # plt.grid(True)
    # sns.swarmplot(data=instance_df, x=x_name, y=y_axis_name, palette=my_pal)
    # plt.savefig(instance_src_path / "pd_dice_swarm.png", dpi=600)
    # plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    # plt.close()
    #
    # plt.grid(True)
    # sns.violinplot(data=instance_df, x=x_name, y=y_axis_name, palette=my_pal, bw=bw, cut=0)
    # sns.swarmplot(data=instance_df, x=x_name, y=y_axis_name, palette=dark_pal, size=marker_size)
    # plt.savefig(instance_src_path / "pd_dice_violin_swarm.png", dpi=600)
    # plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    # plt.close()
    #
    # plt.grid(True)
    # sns.violinplot(data=instance_df, x=x_name, y=y_axis_name, palette=my_pal, bw=bw, cut=0)
    # sns.stripplot(data=instance_df, x=x_name, y=y_axis_name, palette=dark_pal, size=marker_size)
    # plt.savefig(instance_src_path / "pd_dice_violin_strip.png", dpi=600)
    # plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    # plt.close()

    '''
    # GT dice
    instance_df[y_axis_name] = instance_df["mean_gt_dice"]
    plt.grid(True)
    sns.violinplot(data=instance_df, x="dataset", y=y_axis_name, palette=my_pal)
    plt.savefig(instance_src_path / "gt_dice_violin.png", dpi=600)
    plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    plt.close()
   
    plt.grid(True)
    sns.swarmplot(
        data=instance_df, x="dataset", y=y_axis_name, palette=my_pal, size=marker_size
        )
    plt.savefig(instance_src_path / "gt_dice_swarm.png", dpi=600)
    plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    plt.close()
    
    plt.grid(True)
    sns.violinplot(data=instance_df, x="dataset", y=y_axis_name, palette=my_pal)
    sns.swarmplot(
        data=instance_df, x="dataset", y=y_axis_name, palette=dark_pal, size=marker_size
        )
    plt.savefig(instance_src_path / "gt_dice_violin_swarm.png", dpi=600)
    plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    plt.close()

    plt.grid(True)
    sns.violinplot(data=instance_df, x="dataset", y=y_axis_name, palette=my_pal)
    sns.stripplot(
        data=instance_df, x="dataset", y=y_axis_name, palette=dark_pal, size=marker_size
        )
    plt.savefig(instance_src_path / "gt_dice_violin_strip.png", dpi=600)
    plt.title("Mean DICE coefficient (lesion-wise): for CE lesions")
    plt.close()
    '''

if __name__ == "__main__":
    main()