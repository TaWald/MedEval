from matplotlib import gridspec
import seaborn as sns

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
from sem2ins.visualizations.architecture_wise_performance import add_median_labels

MODEL_MAPPING = {
    "task561_mmm_15": "Setting A",
    "task570_mss": "Setting C",
    "task590_mms_predsTs": "Setting A",
    "task570_mss_predsTs": "Setting C",
    "task561_mmm_predsTs": "Setting A",
    "task610_msm_predsTs": "Setting C",
}

DATASET_MAPPING = {
    "test_set_SPACE": "SPACE",
    "test_set_MPRAGE": "MPRAGE",
    "stanford_bm": "Stanford",
    "institutional_hd_bm": "HD Uni",
    "bm_external": "HD Thorax",
    "brain_tr_gammaknife": "GammaKnife",
    # "brainmets_molab": "Molab",
}
color = "#00BA38"
my_pal = [
    "#619CFF",
    "#00BA38",
]  # Trainset palette
oc = [
    "#F8766D",  # Used
    "#D39200",
    "#93AA00",
    "#00BA38",  # Used
    "#00C19F",
    "#00B9E3",
    "#619CFF",  # Used
    "#DB72FB",
]
other_palette = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#F9C74F",
    "#90BE6D",
    "#F94144",
    "#98c1d9",
    "#ffddd2",
    "#9cb380",
]


FACECOLOR = "#eaeaea"


def calculate_wilcoxon(df: pd.DataFrame, metric: str) -> float:
    values_a = df[df["model"] == "Setting A"][metric].values
    values_a = values_a[~np.isnan(values_a)]  # The NaNs are not good. Either set them to 1 or 0. But NaN is bad! 
    values_c = df[df["model"] == "Setting C"][metric].values
    return wilcoxon(values_a, values_c)[1]


def plot_casewise_prec_recall_f1(df: pd.DataFrame):
    # sns.set_style("whitegrid")
    # sns.set_theme("paper", style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 1.5})

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3)

    palette = ["#9DB2BF", "#E23E57"]
    stripplot_kwargs = {
        "x": "dataset",
        "data": df,
        "hue": "model",
        "linewidth": 0.5,
        "size": 2,
        "edgecolor": "black",
        "legend": False,
        "palette": palette,
        "dodge": True,
    }
    boxplot_kwargs = {
        "x": "dataset",
        "data": df,
        "hue": "model",
        "showfliers": False,
        "palette": palette,
        "linewidth": 1,
        # "showmeans": True,
        # "meanprops": {"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 10},
    }

    def rot_xticks(ax):
        ax.tick_params(axis="x", rotation=20, labelright=True)

    def add_vline(ax):
        ax.vlines(3.5, 0, 1, colors="black", linestyles="dashed")

    # -------------------------------- Upper PLots ------------------------------- #
    ax = plt.subplot(gs[0, 0])
    sns.boxplot(y="case_f1", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_f1", ax=ax, **stripplot_kwargs)
    f1_pval = calculate_wilcoxon(df, "case_f1")

    ax.set_title("Instance Detection F1 Score")
    ax.set_ylabel("Instance Detection F1 Score")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    boxplot_kwargs["legend"] = False
    stripplot_kwargs["legend"] = False
    ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    # Now remove legend from following boxplots
    # boxplot_kwargs["legend"] = False

    ax = plt.subplot(gs[0, 1])
    sns.boxplot(y="case_recall", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_recall", ax=ax, **stripplot_kwargs)
    rc_pval = calculate_wilcoxon(df, "case_recall")
    ax.set_title("Instance Detection Sensitivity")
    ax.set_ylabel("Instance Detection Sensitivity")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    if boxplot_kwargs["legend"]:
        ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[0, 2])
    sns.boxplot(y="case_precision", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_precision", ax=ax, **stripplot_kwargs)
    pc_pval = calculate_wilcoxon(df, "case_precision")
    ax.set_title("Instance Detection Precision")
    ax.set_ylabel("Instance Detection Precision")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    if boxplot_kwargs["legend"]:
        ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")
    # ------------------------- Lower instance-wise plots ------------------------ #
    ax = plt.subplot(gs[1, 0])
    sns.stripplot(y="lw_dice", ax=ax, **stripplot_kwargs)
    sns.boxplot(y="lw_dice", ax=ax, **boxplot_kwargs)
    lwdice_pval = calculate_wilcoxon(df, "lw_dice")

    ax.set_title("Instance Voxel DICE")
    ax.set_ylabel("Instance Voxel DICE")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    if boxplot_kwargs["legend"]:
        ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[1, 1])
    sns.stripplot(y="lw_recall", ax=ax, **stripplot_kwargs)
    sns.boxplot(y="lw_recall", ax=ax, **boxplot_kwargs)
    lwdice_pval = calculate_wilcoxon(df, "lw_recall")
    ax.set_ylabel("Instance Voxel Sensitivity")  # Set the y_label to F1-Score
    ax.set_title("Instance Voxel Sensitivity")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    if boxplot_kwargs["legend"]:
        ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[1, 2])
    sns.stripplot(y="lw_precision", ax=ax, **stripplot_kwargs)
    sns.boxplot(y="lw_precision", ax=ax, **boxplot_kwargs)
    lwdice_pval = calculate_wilcoxon(df, "lw_precision")
    ax.set_title("Instance Voxel Precision")
    ax.set_ylabel("Instance Voxel Precision")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.4)
    if boxplot_kwargs["legend"]:
        ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    plt.savefig(f"instance_metrics.pdf")
    plt.savefig(f"instance_metrics.png", dpi=300)


def main():
    src_path = Path("/mnt/cluster-data-all/t006d/nnunetv2/test_data_folder")
    datasets = [
        "stanford_bm",
        "brain_tr_gammaknife",
        # "brainmets_molab",
        "institutional_hd_bm",
        "bm_external",
    ]
    models = ["task561_mmm_15", "task570_mss"]
    spc_dataset = ["test_set_SPACE"]
    spc_models = ["task590_mms_predsTs", "task570_mss_predsTs"]
    mpr_dataset = ["test_set_MPRAGE"]
    mpr_models = ["task561_mmm_predsTs", "task610_msm_predsTs"]

    lwcw = "GTid1_PDid1_eval/instancewise_evaluation/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3/hard_eval/lw_cw_values.csv"
    lwdw = (
        "GTid1_PDid1_eval/instancewise_evaluation/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3/hard_result.csv"
    )

    casewise_result = []
    datasetwise_result = []
    for mod, datasets in [(models, datasets), (spc_models, spc_dataset), (mpr_models, mpr_dataset)]:
        for m in mod:
            for ds in datasets:
                assert src_path.exists()
                assert (src_path / ds).exists()
                assert (src_path / ds / m).exists()
                cw_df = pd.read_csv(src_path / ds / m / lwcw)
                dw_df = pd.read_csv(src_path / ds / m / lwdw)
                cw_df.drop(columns=["Unnamed: 0"], inplace=True)
                dw_df.drop(columns=["Unnamed: 0"], inplace=True)
                cw_df["dataset"] = DATASET_MAPPING[ds]
                dw_df["dataset"] = DATASET_MAPPING[ds]
                cw_df["model"] = MODEL_MAPPING[m]
                dw_df["model"] = MODEL_MAPPING[m]
                casewise_result.append(cw_df)
                datasetwise_result.append(dw_df)

    cw_result = pd.concat(casewise_result, ignore_index=True)
    dw_result = pd.concat(datasetwise_result, ignore_index=True)
    plot_casewise_prec_recall_f1(cw_result)
    # plot_casewise_prec_recall_f1(dw_result, title="Dataset-wide aggregated")


if __name__ == "__main__":
    main()
