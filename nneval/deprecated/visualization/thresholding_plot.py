import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def main():
    src_path = Path(
        "/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/05_04_2022_EVAL/ce_evaluation/instance_evaluation/"
    )
    result_path = (
        src_path
        / "matching_result/train_results/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3/case_wise_stats.csv"
    )
    out_path = Path(
        "/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/05_04_2022_EVAL/visualisierungen"
    )

    result_df = pd.read_csv(result_path)

    x_name = "Volume threshold [mm³]"
    y_f1_name = "F1 Score"
    y_sensitivity = "Sensitivity"
    y_false_positive_per_exam = "No. of FP per exam"
    y_false_negative_rate = "False negative rate"
    y_true_positive_rate = "True positive rate"

    y_precision = "Precision"
    color = "#00BA38"
    my_pal = ["#00BA38", "#F8766D", "#619CFF"]  # Trainset palette
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

    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12}
    )
    
    result_df[y_f1_name] = result_df["mean_f1_score"]
    result_df[x_name] = result_df["filter_size"]
    result_df[y_sensitivity] = result_df["mean_recall"]
    result_df[y_precision] = result_df["mean_precision"]
    result_df[y_false_positive_per_exam] = (
        result_df["n_false_positives"] / result_df["n_cases"]
    )
    result_df[y_false_negative_rate] = (
        result_df["n_false_negatives"] / result_df["n_positives"]
    )
    result_df[y_true_positive_rate] = (
        result_df["n_true_positives"] / result_df["n_positives"]
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot top left (Lesion Dice)
    axs[0].grid(True, color="w")
    axs[0].set_axisbelow(True)
    axs[0].set(facecolor="#eaeaea")
    sns.lineplot(
        data=result_df, x=x_name, y=y_f1_name, color="g", ax=axs[0], label="F1 Score"
    )
    axs[0].set_xlabel("Threshold [mm³]")
    axs[0].set_title("CE F1-Scores for Volume Thresholds")
    axs[0].legend()

    # Plot top right (Lesion Sensitivity)
    axs[1].grid(True, color="w")
    axs[1].set_axisbelow(True)
    axs[1].set(facecolor="#eaeaea")
    twin_ax = axs[1].twinx()
    sns.lineplot(
        data=result_df,
        x=x_name,
        y=y_false_negative_rate,
        color="r",
        ax=axs[1],
        label=y_false_negative_rate,
    )
    sns.lineplot(
        data=result_df,
        x=x_name,
        y=y_true_positive_rate,
        color="b",
        ax=axs[1],
        label=y_true_positive_rate,
    )
    sns.lineplot(
        data=result_df,
        x=x_name,
        y=y_false_positive_per_exam,
        color="k",
        ax=twin_ax,
        label=y_false_positive_per_exam,
    )
    # twin_ax.legend(labels=["False positive per exam"])
    # axs[1].legend(labels=["False negative rate", "True positive rate"])
    lines, labels = twin_ax.get_legend_handles_labels()
    lines2, labels2 = axs[1].get_legend_handles_labels()
    axs[1].get_legend().remove()
    twin_ax.legend(lines + lines2, labels + labels2,)

    axs[1].set_xlabel("Threshold [mm³]")
    axs[1].set_title("CE lesions")
    axs[1].set_ylabel("True positive rate / False negative rate")
    plt.savefig(str(out_path / "f1_score_volume_thresholds_300dpi.tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "f1_score_volume_thresholds_600dpi.tiff"), dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "f1_score_volume_thresholds_1200dpi.tiff"), dpi=1200, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "f1_score_volume_thresholds_1200dpi.eps"), dpi=1200)


if __name__ == "__main__":
    main()
