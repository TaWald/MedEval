import seaborn as sns

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd



def main():
    src_path = Path(
        "/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/11_02_2022_EVAL/ce_evaluation/instance_evaluation/matching_result"
        )
    p1 = ["train_results", "test_results", "independent_test_results"]
    p2 = "gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3"
    file = "case_wise_tp_fp_fn_no_filter.csv"
    
    file_paths = [src_path / p / p2 / file for p in p1]
    train_pd = pd.read_csv(file_paths[0])
    test_pd = pd.read_csv(file_paths[1])
    indtest_pd = pd.read_csv(file_paths[2])
    
    train_pd["dataset"] = "TR"
    test_pd["dataset"] = "TS"
    indtest_pd["dataset"] = "ITS"
    
    joint_pd = pd.concat([train_pd, test_pd, indtest_pd])
    
    title = "F1 score per patient: for CE lesions"
    y_axis_name = "F1-Score"
    linewidth = .5
    marker_size = 3
    joint_pd[y_axis_name] = joint_pd["f1_score"]
    # sns.set_theme("paper", style="whitegrid")
    sns.set(rc={'axes.facecolor': "#eaeaea"})
    my_pal = ["#00BA38",  "#F8766D", "#619CFF"]
    dark_pal = {"TR": "darkgreen", "TS": "darkred", "ITS": "darkblue"}
    
    plt.rcParams.update({"font.sans-serif": ["Arial"],
                                "font.size": 10})
    
    plt.grid(True)
    sns.violinplot(data=joint_pd, x="dataset", y=y_axis_name, palette=my_pal)
    plt.savefig(src_path / "f1_violin.tiff", dpi=1200)
    plt.title(title)
    plt.close()
    
    # Strip
    plt.grid(True)
    sns.stripplot(data=joint_pd, x="dataset", y=y_axis_name, palette=my_pal, linewidth=linewidth, size=marker_size)
    plt.savefig(src_path / "f1_strip.tiff", dpi=1200)
    plt.title(title)
    plt.close()
    
    # Swarm
    plt.grid(True)
    sns.swarmplot(data=joint_pd, x="dataset", y=y_axis_name, palette=my_pal, linewidth=linewidth, size=marker_size)
    plt.savefig(src_path / "f1_swarm.tiff", dpi=1200)
    plt.title(title)
    plt.close()
    
    
    # Violin & strip
    plt.grid(True)
    sns.violinplot(data=joint_pd, x="dataset", y=y_axis_name, palette=my_pal)
    sns.stripplot(data=joint_pd, x="dataset", y=y_axis_name, palette=dark_pal, size=marker_size,
                  linewidth=linewidth)
    plt.savefig(src_path / "f1_violin_strip.tiff", dpi=1200)
    plt.title(title)
    plt.close()

    # Violin & swarm
    plt.grid(True)
    sns.violinplot(data=joint_pd, x="dataset", y=y_axis_name, palette=my_pal)
    sns.swarmplot(
        data=joint_pd, x="dataset", y=y_axis_name, palette=dark_pal, size=marker_size,
        linewidth=linewidth
        )
    plt.savefig(src_path / "f1_violin_swarm.tiff", dpi=1200)
    plt.title(title)
    plt.close()


if __name__ == "__main__":
    main()