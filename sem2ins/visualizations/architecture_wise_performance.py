import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects


def add_median_labels(ax: plt.Axes, fmt: str = ".1f", y_offset: float = 0, fontscale: float = 1) -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    lines_per_box = len(lines) // len(boxes)
    for median in lines[4::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(
            x,
            y + y_offset,
            f"{value:{fmt}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=12 * fontscale,
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )


f1_scores = {
    "mprage": {
        "3d_densenet": [0.8171, 0.6829, 0.8049, 0.6463, 0.8293],
        "3d_resnet_50": [0.7317, 0.6707, 0.6951, 0.6951, 0.7683],
        "3d_resnet_18": [0.8415, 0.7439, 0.7927, 0.7683, 0.8293],
        "3d_efficientnet_b0": [0.6463, 0.7195, 0.8415, 0.7317, 0.8293],
    },
    "space": {
        "3d_densenet": [0.8293, 0.8415, 0.8659, 0.8902, 0.8049],
        "3d_resnet_50": [0.9268, 0.8780, 0.8293, 0.8537, 0.9024],
        "3d_resnet_18": [0.9390, 0.9024, 0.8902, 0.9512, 0.9207],
        "3d_efficientnet_b0": [0.7561, 0.7317, 0.7317, 0.8171, 0.8049],
    },
    "mprage_3_in": {
        "3d_densenet": [0.7805, 0.7683, 0.7439, 0.8171, 0.8293],
        "3d_resnet_50": [0.8171, 0.8049, 0.8293, 0.7439, 0.6829],
        "3d_resnet_18": [0.7439, 0.8049, 0.7439, 0.7683, 0.7927],
        "3d_efficientnet_b0": [0.7805, 0.7073, 0.6951, 0.7927, 0.7439],
    },
}

architecture_mapping = {
    "3d_densenet": "DenseNet121",
    "3d_resnet_50": "ResNet50",
    "3d_resnet_18": "ResNet18",
    "3d_efficientnet_b0": "EfficientNet B0",
}

input_data_mapping = {
    "mprage": "MPRAGE (late contrast)",
    "space": "SPACE",
    "mprage_3_in": "MPRAGE (all contrasts)",
}


def load_f1_scores_as_df():
    df = pd.DataFrame.from_dict(f1_scores)
    df["Architecture"] = df.index
    df["Architecture"] = df["Architecture"].replace(architecture_mapping)
    df = df.explode(["mprage", "space", "mprage_3_in"])
    df.rename(
        columns=input_data_mapping,
        inplace=True,
    )
    df = pd.melt(df, id_vars=["Architecture"], var_name="Sequence", value_name="F1 Score")
    desired_order = ["MPRAGE (late contrast)", "MPRAGE (all contrasts)", "SPACE"]
    df["Sequence"] = pd.Categorical(df["Sequence"], desired_order, ordered=True)

    return df


if __name__ == "__main__":
    # Plot the results
    pd_df = load_f1_scores_as_df()
    plt.figure(figsize=(10, 6))
    palette = ["#526D82", "#9DB2BF", "#E23E57"]
    ax = sns.boxplot(data=pd_df, x="Architecture", y="F1 Score", hue="Sequence", palette=palette)
    sns.stripplot(
        data=pd_df,
        x="Architecture",
        y="F1 Score",
        hue="Sequence",
        palette=palette,
        dodge=True,
        linewidth=1,
        jitter=True,
        marker="o",
        alpha=1,
        legend=False,
        edgecolor="white",
    )
    ax.set_facecolor("#eaeaea")
    # Add annotations for the median values
    ax.legend(title="Input Sequence")
    add_median_labels(ax, fmt=".1%", y_offset=0, fontscale=0.5)
    plt.title("Classification F1 Scores of different Architectures.")
    plt.grid(axis="y")
    plt.savefig("architecture_wise_performance.pdf")
    plt.savefig("architecture_wise_performance.png", dpi=300)
    plt.close()

    print(0)
