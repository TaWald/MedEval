from functools import partial
from matplotlib import gridspec
import seaborn as sns
from itertools import combinations

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
from nneval.visualizations.architecture_wise_performance import add_median_labels

from statannotations.Annotator import Annotator

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
    "stanford_bm": "SUSM",
    # "institutional_hd_bm": "HD Uni",
    # "bm_external": "HD Thorax",
    "brain_tr_gammaknife": "UMMC",
    "all_institutional_hd_bm": "UKHD",
    "yale_bm": "YNHH",
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
        "size": 2.5,
        "edgecolor": "black",
        # "legend": False,
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

    def add_statistical_annotation(ax: plt.Axes, df: pd.DataFrame, metric: str):
        box_content = list(
            [
                ((ds, m1), (ds, m2))
                for ds in df["data"]["dataset"].unique()
                for m1, m2 in combinations(df["data"]["model"].unique(), 2)
            ]
        )
        annotator = Annotator(
            ax,
            box_content,
            data=df["data"],
            x=df["x"],
            hue=df["hue"],
            y=metric,
        )
        annotator = annotator.configure(
            test="Wilcoxon",
            loc="inside",
            verbose=2,
            hide_non_significant=True
            # text_format="star"
            # pvalue_format={
            #     "pvalue_format_string": "{:.1e}",
            #     "text_format": "simple",
            #     "show_test_name": False,
            # },
            # show_test_name=False,
            # test_short_name=False,
        )
        annotator.apply_and_annotate()
        return

    def rot_xticks(ax):
        ax.tick_params(axis="x", rotation=20, labelright=True)

    def add_vline(ax):
        ax.vlines(3.5, 0, 1, colors="black", linestyles="dashed")

    # -------------------------------- Upper PLots ------------------------------- #
    ax = plt.subplot(gs[0, 0])

    sns.boxplot(y="case_f1", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_f1", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "case_f1")
    ax.set_title("Instance Detection F1 Score")
    ax.set_ylabel("Instance Detection F1 Score")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # boxplot_kwargs["legend"] = False
    # stripplot_kwargs["legend"] = False
    ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    # Now remove legend from following boxplots
    # boxplot_kwargs["legend"] = False

    ax = plt.subplot(gs[0, 1])
    sns.boxplot(y="case_recall", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_recall", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "case_recall")
    ax.set_title("Instance Detection Sensitivity")
    ax.set_ylabel("Instance Detection Sensitivity")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # if boxplot_kwargs.get("legend", False):
    #     ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[0, 2])
    sns.boxplot(y="case_precision", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="case_precision", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "case_precision")
    ax.set_title("Instance Detection PPV")
    ax.set_ylabel("Instance Detection PPV")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # if boxplot_kwargs["legend"]:
    #     ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")
    # ------------------------- Lower instance-wise plots ------------------------ #
    ax = plt.subplot(gs[1, 0])
    sns.boxplot(y="lw_dice", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="lw_dice", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "lw_dice")

    ax.set_title("Instance Voxel DICE")
    ax.set_ylabel("Instance Voxel DICE")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # if boxplot_kwargs["legend"]:
    #     ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[1, 1])
    sns.boxplot(y="lw_recall", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="lw_recall", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "lw_recall")
    ax.set_ylabel("Instance Voxel Sensitivity")  # Set the y_label to F1-Score
    ax.set_title("Instance Voxel Sensitivity")
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # if boxplot_kwargs["legend"]:
    #     ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    ax = plt.subplot(gs[1, 2])
    sns.boxplot(y="lw_precision", ax=ax, **boxplot_kwargs)
    sns.stripplot(y="lw_precision", ax=ax, **stripplot_kwargs)
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.6)
    add_statistical_annotation(ax, boxplot_kwargs, "lw_precision")
    ax.set_title("Instance Voxel PPV")
    ax.set_ylabel("Instance Voxel PPV")  # Set the y_label to F1-Score
    ax.set_xlabel("Test Dataset")
    rot_xticks(ax)
    add_vline(ax)
    # if boxplot_kwargs["legend"]:
    #     ax.legend(title="Model trained on")
    ax.set_facecolor(FACECOLOR)
    ax.grid(axis="y")

    plt.savefig(f"instance_metrics.pdf")
    plt.savefig(f"instance_metrics.png", dpi=300)


def load_dataset_case_wise_results(src_path: Path, dataset: str, model: str):
    """
    This loads the results from a single dataset and model.
    :param src_path: The path to the root folder that contains the results.
    :param dataset: The dataset you want to load.
    :param model: The model you want to load
    """
    assert src_path.exists(), "The source path does not exist"
    assert (src_path / dataset).exists(), "The dataset does not exist"
    assert (src_path / dataset / model).exists(), "The model does not exist"
    default_config = "GTid1_PDid1_eval/instancewise_evaluation/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3"
    lwcw_path = "hard_eval/lw_cw_values.csv"
    lwdw_path = "hard_result.csv"
    lwcw_df = pd.read_csv(src_path / dataset / model / default_config / lwcw_path)
    dw_df = pd.read_csv(src_path / dataset / model / default_config / lwdw_path)
    lwcw_df.drop(columns=["Unnamed: 0"], inplace=True)
    dw_df.drop(columns=["Unnamed: 0"], inplace=True)
    lwcw_df["dataset"] = DATASET_MAPPING[dataset]
    dw_df["dataset"] = DATASET_MAPPING[dataset]
    lwcw_df["model"] = MODEL_MAPPING[model]
    dw_df["model"] = MODEL_MAPPING[model]
    return lwcw_df, dw_df


def load_all_results(src_path: Path, data_models: list[tuple[list[str], list[str]]]):
    """
    Loads the results from more than one dataset and model and concatenates them into one dataframe.
    :param src_path: The path to the root folder that contains the results.
    :param data_models: A list of tuples that contains the combinations of interest `(dataset, model)` you want to load.
    """
    casewise_result = []
    datasetwise_result = []
    for models, datasets in data_models:
        for model in models:
            for ds in datasets:
                cw_df, dw_df = load_dataset_case_wise_results(src_path, ds, model)
                casewise_result.append(cw_df)
                datasetwise_result.append(dw_df)
    cw_result = pd.concat(casewise_result, ignore_index=True)
    dw_result = pd.concat(datasetwise_result, ignore_index=True)
    return cw_result, dw_result


def _load_results_of_interest():
    src_path = Path("/mnt/cluster-data-all/t006d/nnunetv2/test_data_folder")
    datasets = [
        "stanford_bm",
        "yale_bm",
        "brain_tr_gammaknife",
        # "brainmets_molab",
        # "institutional_hd_bm",
        "all_institutional_hd_bm",
        # "bm_external",
    ]
    models = ["task561_mmm_15", "task570_mss"]
    spc_dataset = ["test_set_SPACE"]
    spc_models = ["task590_mms_predsTs", "task570_mss_predsTs"]
    mpr_dataset = ["test_set_MPRAGE"]
    mpr_models = ["task561_mmm_predsTs", "task610_msm_predsTs"]

    lwcw_res, lwdw_res = load_all_results(
        src_path=src_path,
        data_models=[(models, datasets), (spc_models, spc_dataset), (mpr_models, mpr_dataset)],
    )
    return lwcw_res, lwdw_res


def main():
    lwcw_res, lwdw_res = _load_results_of_interest()
    plot_casewise_prec_recall_f1(lwcw_res)
    # plot_casewise_prec_recall_f1(dw_result, title="Dataset-wide aggregated")


if __name__ == "__main__":
    main()
