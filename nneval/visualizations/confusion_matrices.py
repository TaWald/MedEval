from functools import partial
from matplotlib import gridspec
import seaborn as sns
from itertools import combinations

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

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


def plot_confusion_matrices(df: pd.DataFrame):
    # sns.set_style("whitegrid")
    # sns.set_theme("paper", style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 1.5})

    unique_datasets = df["dataset"].unique()
    all_results = []
    for unique_dataset in unique_datasets:
        unique_ds_df = df[df["dataset"] == unique_dataset]
        unique_models = unique_ds_df["model"].unique()

        for unique_model in unique_models:
            uq_ds_model_df = unique_ds_df[unique_ds_df["model"] == unique_model]
            tps = np.nansum(uq_ds_model_df["true_positive"])
            fns = np.nansum(uq_ds_model_df["false_negative"])
            fps = np.nansum(1 - uq_ds_model_df["true_positive"])
            all_results.append(
                {
                    "dataset": unique_dataset,
                    "model": "Setting B" if unique_model == "Setting C" else unique_model,
                    "TP": tps,
                    "FN": fns,
                    "FP": fps,
                }
            )

    # Define colors and annotations
    colors = {"Setting A": "gray", "Setting B": "red"}

    fig, axes = plt.subplots(nrows=len(unique_datasets), ncols=2, figsize=(10, 18), dpi=300)

    df = pd.DataFrame(all_results)
    for i, dataset in enumerate(unique_datasets):
        for j, model in enumerate(["Setting A", "Setting B"]):
            ax = axes[i, j]
            sub_df = df[(df["dataset"] == dataset) & (df["model"] == model)]

            # Confusion matrix values
            TP = sub_df["TP"].values[0]
            FN = sub_df["FN"].values[0]
            FP = sub_df["FP"].values[0]

            # Create confusion matrix array
            matrix = np.array([[TP, FP], [FN, 0]])

            # Plot confusion matrix
            cmap = "Grays" if model == "Setting A" else "Reds"
            c = ax.matshow(matrix, cmap=cmap, alpha=0.75)
            for (x, y), value in np.ndenumerate(matrix):
                ax.text(
                    y,
                    x,
                    f'{"N/A" if value == 0 else value}',
                    va="center",
                    ha="center",
                    color="black",
                    fontweight="bold",
                )

            # Aesthetics
            ax.set_yticklabels(["Pred Pos", "Pred Neg"])
            ax.set_xticklabels(["Actual Pos", "Actual Neg"])
            # ax.set_xlabel(f'{model}\n{dataset}')
            ax.xaxis.set_label_position("top")
            ax.set_title(f"{model} on {dataset}")

            if i == len(unique_datasets) - 1:
                ax.set_xticks([0, 1])
            else:
                ax.set_xticks([0, 1])

            if j == 0:
                ax.set_yticks([0, 1])
            else:
                ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Save the figure as a PDF
    plt.savefig("confusion_matrices.pdf")
    plt.close()


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
    _, lwdw_res = _load_results_of_interest()
    plot_confusion_matrices(lwdw_res)


if __name__ == "__main__":
    main()
