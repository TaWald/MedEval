from typing import Sequence

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def main():
    path_inst_test = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/Task067_BrainMets_noT2/eval_test/GTid2_PDid2_eval/instancewise_evaluation/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3")
    path_ext_test = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/Task142_ThoraxKlinik/eval_ext/GTid2_PDid2_eval/instancewise_evaluation/gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3")
    
    inst_df = pd.read_csv(path_inst_test/"all_groundtruth_instances.csv")
    ext_df = pd.read_csv(path_ext_test / "all_groundtruth_instances.csv")
    
    out_path = Path("/home/tassilowald/Data/Datasets/BRAINMETASTASIS_PROJECT/05_04_2022_EVAL/visualisierungen")
    
    x_name = "log(Volume) [Vol in cm³]"
    y_name = "Dice coefficient"
    inst_df[y_name] = inst_df["max_dice"]
    inst_df[x_name] = np.log(inst_df["groundtruth_size"]/1000.)
    ext_df[y_name] = ext_df["max_dice"]
    ext_df[x_name] = np.log(ext_df["groundtruth_size"]/1000.)
    
    inst_dice = inst_df[y_name].to_numpy()
    inst_vol = inst_df[x_name].to_numpy()
    
    ext_dice = ext_df[y_name].to_numpy()
    ext_vol = ext_df[x_name].to_numpy()
    
    inst_corr = spearmanr(inst_dice, inst_vol)
    ext_corr = spearmanr(ext_dice, ext_vol)
    
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12}
    )
    
    color = "#00BA38"
    my_pal = ["#00BA38", "#F8766D", "#619CFF"]  # Trainset palette
    oc = ["#F8766D",  # Used
          "#D39200",
          "#93AA00",
          "#00BA38",  # Used
          "#00C19F",
          "#00B9E3",
          "#619CFF",  # Used
          "#DB72FB"]
    axs:Sequence[plt.Axes]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot top left (Lesion Dice)
    axs[0].grid(True, color="w")
    axs[0].set_axisbelow(True)
    axs[0].set(facecolor="#eaeaea")
    sns.regplot(data=inst_df, x=x_name, y=y_name, color="#F8766D", ax=axs[0])
    axs[0].set_title("Dice to log(Volume) Correlation (Institutional test set)")
    axs[0].set_ylim(0, 1.)
    # axs[0].set_xlim(0, 20.)
    axs[0].text(axs[0].get_xlim()[1], 0,
                f"Spearman Rank Correlation: {inst_corr[0]:.03f}\n{'p<0.0001' if inst_corr[1]<0.0001 else inst_corr[1].format(':03f')}",
                ha="right", va="bottom")
    
    # Plot top right (Lesion Sensitivity)
    axs[1].grid(True, color="w")
    axs[1].set_axisbelow(True)
    axs[1].set(facecolor="#eaeaea")
    sns.regplot(data=ext_df, x=x_name, y=y_name, color="#619CFF", ax=axs[1])
    axs[1].set_ylim(0, 1.)
    # axs[1].set_xlim(0, 20.)
    # twin_ax.legend(labels=["False positive per exam"])
    # axs[1].legend(labels=["False negative rate", "True positive rate"])
    axs[1].text(axs[1].get_xlim()[1], 0, f"Spearman Rank Correlation: {ext_corr[0]:.03f}\n{'p<0.0001' if ext_corr[1]<0.0001 else ext_corr[1].format(':03f')}",
                ha="right", va="bottom")
    axs[1].set_title("Dice to log(Volume) Correlation (External test set)")
    plt.savefig(str(out_path / "dice_to_volume_300dpi.tiff"), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "dice_to_volume_600dpi.tiff"), dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "dice_to_volume_1200dpi.tiff"), dpi=1200, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(str(out_path / "dice_to_volume_1200dpi.eps"), dpi=1200)
    
    
    # format="tiff", pil_kwargs={"compression": "tiff_lzw"}


if __name__ == "__main__":
    main()