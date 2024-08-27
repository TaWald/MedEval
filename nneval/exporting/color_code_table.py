import pandas as pd
from matplotlib import colors
import numpy as np


# 256 to [0,1]
def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


# [0,1] to 256
def infer_to_256(x):
    return int(np.interp(x=x, xp=[0, 1], fp=[0, 255]))


def build_custom_continuous_cmap(*rgb_list):
    """
    Generating any custom continuous colormap, user should supply a list of (R,G,B) color taking the value from [0,255], because this is
    the format the adobe color will output for you.

    Examples::

        test_cmap = build_custom_continuous_cmap([64,57,144],[112,198,162],[230,241,146],[253,219,127],[244,109,69],[169,23,69])
        fig,ax = plt.subplots()
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=diverge_cmap),ax=ax)

    .. image:: ./_static/custom_continuous_cmap.png
        :height: 400px
        :width: 550px
        :align: center
        :target: target

    """
    all_red = []
    all_green = []
    all_blue = []
    for rgb in rgb_list:
        all_red.append(rgb[0])
        all_green.append(rgb[1])
        all_blue.append(rgb[2])
    # build each section
    n_section = len(all_red) - 1
    red = tuple([(1 / n_section * i, inter_from_256(v), inter_from_256(v)) for i, v in enumerate(all_red)])
    green = tuple([(1 / n_section * i, inter_from_256(v), inter_from_256(v)) for i, v in enumerate(all_green)])
    blue = tuple([(1 / n_section * i, inter_from_256(v), inter_from_256(v)) for i, v in enumerate(all_blue)])
    cdict = {"red": red, "green": green, "blue": blue}
    new_cmap = colors.LinearSegmentedColormap("new_cmap", segmentdata=cdict)
    return new_cmap


def plot_nsd_table(df: pd.DataFrame) -> None:
    table = df.drop(axis=1, labels=["Unnamed: 7", "AVERAGE", "RANK"])

    table[["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"]] = (
        table[["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"]]
        .apply(lambda col: pd.to_numeric(col, errors="coerce"), axis=1)
        .round(decimals=2)
    )
    table = table.reindex(
        labels=[
            "nnU-Net (org)",
            "nnU-Net ResEnc M",
            "nnU-Net ResEnc L",
            "nnU-Net ResEnc XL",
            "MedNeXt k3",
            "MedNeXt k5",
            "STU-Net S",
            "STU-Net B",
            "STU-Net L",
            "SwinUNETR",
            "SwinUNETRV2",
            "nnFormer",
            "CoTr",
            "No-Mamba Base",
            "U-Mamba Bot",
            "U-Mamba Enc",
            "A3DS SegResNet",
            "A3DS DiNTS",
            "A3DS SwinUNETR ",
        ],
        axis=0,
    )

    print(table[["BTCV"]])
    table.style.background_gradient(
        cmap,
        subset=["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"],
        axis=0,
    ).format(precision=2).to_latex(
        "~/Downloads/nsd_table_out.tex", column_format="lcccccc", hrules=True, convert_css=True
    )


def plot_sd_table(df: pd.DataFrame) -> None:
    table = df
    for ds in ["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"]:
        table.drop(labels=[(ds, "Dice"), (ds, "Rank")], axis=1, inplace=True)
    table.drop(labels=[("AVERAGE", "Dice"), ("AVERAGE", "Rank")], axis=1, inplace=True)
    table.columns = table.columns.droplevel(1)
    table = table[table.index.notnull()]

    """
    nnUNetTrainer__nnUNetPlans                     0.025731   0.00802  0.034916  0.006152  0.019653   0.00426
    nnUNetTrainer__nnUNetResEncUNetMPlans          0.023814  0.006168  0.026043  0.006677  0.022043  0.005657
    nnUNetTrainer__nnUNetResEncUNetLPlans          0.027153  0.006035  0.023733  0.005715  0.012875  0.005876
    nnUNetTrainer__nnUNetResEncUNetXLPlans         0.026774  0.005078  0.023623  0.006214   0.01212  0.004316
    STUNET small                                   0.021986  0.005988  0.032972   0.00715  0.017035  0.004204
    STUNET base                                    0.022928  0.007784  0.036298    0.0104  0.018981  0.005176
    STUNET large                                    0.02646  0.008469       NaN  0.006175  0.021103  0.004486
    auto3Dseg - dints - 40G                        0.030155  0.022056  0.025209  0.007898  0.053179  0.013417
    auto3Dseg - segresnet - 40G                    0.029708  0.003337  0.027483   0.00524  0.017477  0.004834
    auto3Dseg - swinunetr - 40G                    0.018303  0.035544  0.066063   0.00688  0.014794   0.00642
    nnSwinUNETRTrainer48 (48 feature size)         0.027275  0.006469  0.030947  0.007495   0.02005  0.004447
    nnSwinUNETRTrainerV2 (48 feature size)         0.020728  0.005099   0.02752   0.00552  0.017345   0.00556
    nnFormerLITSTrainer (adapted from other plan)  0.020607  0.002134  0.022642    0.0052  0.041832  0.004979
    nnCoTrTrainer                                  0.027912  0.008264  0.027971  0.006924  0.014151  0.006448
    mednext L k3                                   0.020914  0.002598  0.023381  0.006569  0.009439   0.00429
    mednext L k5 up kernel                         0.019962  0.001989   0.02419  0.005894  0.011736  0.004347
    nnUNetTrainerUMambaBot                         0.022693  0.005918  0.021193  0.007082  0.026526  0.004325
    nnUNetTrainerUMambaEnc_noMamba                 0.023039  0.004728  0.017138  0.006415  0.022447     0.005
    nnUNetTrainerUMambaEnc_lr1en3                  0.018587  0.005139  0.029101   0.00553  0.021412  0.003214
    """
    table = table.rename(
        index={
            "nnUNetTrainer__nnUNetPlans": "nnU-Net (org)",
            "nnUNetTrainer__nnUNetResEncUNetMPlans": "nnU-Net ResEnc M",
            "nnUNetTrainer__nnUNetResEncUNetLPlans": "nnU-Net ResEnc L",
            "nnUNetTrainer__nnUNetResEncUNetXLPlans": "nnU-Net ResEnc XL",
            "STUNET small": "STU-Net S",
            "STUNET base": "STU-Net B",
            "STUNET large": "STU-Net L",
            "auto3Dseg - dints - 40G": "A3DS DiNTS",
            "auto3Dseg - segresnet - 40G": "A3DS SegResNet",
            "auto3Dseg - swinunetr - 40G": "A3DS SwinUNETR",
            "nnSwinUNETRTrainer48 (48 feature size)": "SwinUNETR",
            "nnSwinUNETRTrainerV2 (48 feature size)": "SwinUNETRV2",
            "nnFormerLITSTrainer (adapted from other plan)": "nnFormer",
            "nnCoTrTrainer": "CoTr",
            "mednext L k3": "MedNeXt k3",
            "mednext L k5 up kernel": "MedNeXt k5",
            "nnUNetTrainerUMambaBot": "U-Mamba Bot",
            "nnUNetTrainerUMambaEnc_noMamba": "U-Mamba Enc",
            "nnUNetTrainerUMambaEnc_lr1en3": "No-Mamba Base",
        }
    )

    # table[["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"]] = (
    #     table[["BTCV", "ACDC", "LiTS", "BraTS2021", "KiTS2023", "AMOS2022"]]
    #     .apply(lambda col: pd.to_numeric(col, errors="coerce"), axis=1)
    #     .round(decimals=2)
    # )
    table = table.reindex(
        labels=[
            "nnU-Net (org)",
            "nnU-Net ResEnc M",
            "nnU-Net ResEnc L",
            "nnU-Net ResEnc XL",
            "MedNeXt k3",
            "MedNeXt k5",
            "STU-Net S",
            "STU-Net B",
            "STU-Net L",
            "SwinUNETR",
            "SwinUNETRV2",
            "nnFormer",
            "CoTr",
            "No-Mamba Base",
            "U-Mamba Bot",
            "U-Mamba Enc",
            "A3DS SegResNet",
            "A3DS DiNTS",
            "A3DS SwinUNETR",
        ],
        axis=0,
    )

    print(table[["BTCV"]])
    table.style.format(precision=2, formatter=lambda x: f"{x*100:.2}\%").to_latex(
        "~/Downloads/sd_table_out.tex", column_format="lcccccc", hrules=True, convert_css=True
    )


if __name__ == "__main__":
    cmap = build_custom_continuous_cmap([255, 167, 155], [255, 230, 134], [163, 232, 146])
    # cmap = "RdYlGn"

    nsd_sheet = pd.read_excel(
        "~/Downloads/mamba_slumber_results.xlsx", sheet_name="NSD_at_tol_2mm", header=0, index_col=0
    )
    plot_nsd_table(nsd_sheet)

    df = pd.read_excel(
        "~/Downloads/mamba_slumber_results.xlsx", sheet_name="Summary_results_4_paper", header=(0, 1), index_col=0
    )
    plot_sd_table(df)
