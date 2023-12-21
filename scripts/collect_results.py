import json
import os
from pathlib import Path
import pandas as pd
from pandas import json_normalize
import itertools
import xlsxwriter

SNS_COLORS_PASTEL = [
    "#a1c9f4",
    "#ffb482",
    "#8de5a1",
    "#ff9f9b",
    "#d0bbff",
    "#debb9b",
    "#fab0e4",
    "#cfcfcf",
    "#fffea3",
    "#b9f2f0",
]

PD_MAPPING = {
    "pred200": "Setting A; v2",
    "pred201": "Setting B; v2",
    "pred202": "Setting C; v2",
    "pred203": "Setting D; v2",
    "task561_mmm_predsTs": "Setting A; v1",
    "task590_mms_predsTs": "Setting A; v1",
    "task580_smm_predsTs": "Setting B; v1",
    "task620_sms_predsTs": "Setting B; v1",
    "task610_msm_predsTs": "Setting C; v1",
    "task570_mss_predsTs": "Setting C; v1",
    "task600_ssm_predsTs": "Setting D; v1",
    "task550_sss_predsTs": "Setting D; v1",
    "task561_mmm_15": "Setting A; v1",
    "task570_mss": "Setting C; v1",
}

DS_MAPPING = {
    "bm_external": "ThoraxKlinik",
    "institutional_hd_bm": "HD UniKlinik",
    "stanford_bm": "Stanford BrainMetShare",
    "test_set_SPACE": "Test Set SPACE",
    "test_set_MPRAGE": "Test Set MPRAGE",
    "brain_tr_gammaknife": "GammaKnife",
    "brainmets_molab": "Molab",
}


def aggregate_semantic_means_results(root_dir: Path, ds_pred_pairs: tuple[str, str], coi: int) -> list[dict]:
    all_results = []
    for ds, pd in ds_pred_pairs:
        pth = "GTid{}_PDid{}_eval".format(coi, coi)
        res = "samplewise_evaluation"
        mean_json = "casewise_means_noresample.json"
        json_path = root_dir / ds / pd / pth / res / mean_json
        with json_path.open("r") as f:
            data = json.load(f)[0]
            keys = list(data.keys())
            data["dataset"] = DS_MAPPING[ds]
            data["model"] = PD_MAPPING[pd]
        all_results.append(data)
    return all_results


def aggregate_instance_means_results(root_dir: Path, ds_pred_pairs: tuple[str, str], coi: int) -> list[dict]:
    all_results = []
    for ds, pd in ds_pred_pairs:
        pth = "GTid{}_PDid{}_eval".format(coi, coi)
        res = "instancewise_evaluation"
        config = "gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3"
        hard_res = "hard_eval"
        lw_cw_stats = "lw_cw_values_stats.json"
        json_path = root_dir / ds / pd / pth / res / config / hard_res / lw_cw_stats
        with json_path.open("r") as f:
            data = json.load(f)
            keys = list(data.keys())
            data["dataset"] = DS_MAPPING[ds]
            data["model"] = PD_MAPPING[pd]
        all_results.append(data)
    return all_results


def aggregate_dataset_wide_means_results(root_dir: Path, ds_pred_pairs: tuple[str, str], coi: int) -> list[dict]:
    all_results = []
    for ds, pd in ds_pred_pairs:
        pth = "GTid{}_PDid{}_eval".format(coi, coi)
        res = "instancewise_evaluation"
        config = "gtKernel_ball_Ndilation_3_pdKernel_ball_Ndilation_3"
        hard_res = "hard_eval"
        lw_cw_stats = "dataset_wide_values.json"
        json_path = root_dir / ds / pd / pth / res / config / hard_res / lw_cw_stats
        with json_path.open("r") as f:
            data = json.load(f)
            keys = list(data.keys())
            data["dataset"] = DS_MAPPING[ds]
            data["model"] = PD_MAPPING[pd]
        all_results.append(data)
    return all_results


def create_nice_dfs(loaded_json: dict) -> pd.DataFrame:
    flat_df = json_normalize(loaded_json, sep=".")
    pd_df = pd.DataFrame(flat_df)
    # Set the MultiIndex
    pd_df.set_index(["dataset", "model"], inplace=True)
    multi_col_tuples = [tuple(x) for x in pd_df.columns.str.split(".", expand=False)]
    multi_index = pd.MultiIndex.from_tuples(multi_col_tuples)
    pd_df.columns = multi_index
    return pd_df


def color_rows_by_dataset(worksheet, formats, df):
    # Find the indices where the values change
    n_header = df.columns.nlevels
    first_col_values = [v for v in df.index.get_level_values(0)]
    # Merge the cells in the first row

    color_index = 0
    prev_val = first_col_values[0]
    for cnt, v in enumerate(first_col_values):
        if v != prev_val:
            color_index += 1
            prev_val = v
        worksheet.set_row(cnt + n_header, cell_format=formats[color_index])

    return


def join_header_cells(worksheet, df):
    n_row_indices = df.index.nlevels
    for i in range(df.columns.nlevels - 1):
        first_row_values = [v for v in df.columns.get_level_values(i)]
        # Find the indices where the values change
        value_lengths = [(k, len(list(g))) for k, g in itertools.groupby(first_row_values)]
        # Merge the cells in the first row
        shift = n_row_indices
        for k, l in value_lengths:
            start_index = shift
            end_index = shift + l - 1
            shift += l
            if l == 1:
                worksheet.write(i, start_index, k)
            if pd.notna(k):
                worksheet.merge_range(i, start_index, i, end_index, k)
    last_level = df.columns.nlevels - 1
    last_row_values = [v for v in df.columns.get_level_values(last_level)]
    for j, value in enumerate(last_row_values):
        if pd.notna(value):
            worksheet.write(last_level, j + n_row_indices, value)


def save_pretty_excel(dfs: list[pd.DataFrame], sheet_name: list[str]):
    # Your existing code...

    # Create a list of dictionaries for each row in the DataFrame

    with pd.ExcelWriter("pretty_outputs.xlsx", engine="xlsxwriter") as writer:
        for df, name in zip(dfs, sheet_name):
            workbook = writer.book
            formats = [workbook.add_format({"bg_color": SNS_COLORS_PASTEL[i]}) for i in range(len(SNS_COLORS_PASTEL))]
            df = df.sort_index()
            df.to_excel(writer, sheet_name=name, index=True, header=False, startrow=df.columns.nlevels - 1)

            worksheet = writer.sheets[name]
            join_header_cells(worksheet, df)
            worksheet.merge_range("A1:A2", "Dataset")
            worksheet.merge_range("B1:B2", "Model")

            color_rows_by_dataset(worksheet, formats, df)


def sem_json_to_result(json_results: dict):
    """Flatten the json results to value and columns"""
    all_results = []
    for v in json_results:
        model = v["model"]
        dataset = v["dataset"]
        for kk in ["dice", "precision", "sensitivity"]:
            for kkk in ["mean", "std"]:
                all_results.append(
                    {
                        "category": "semantic",
                        "dataset": dataset,
                        "model": model,
                        "metric": kk + "_" + kkk,
                        "value": v[kk][kkk],
                    }
                )
    return all_results


def lw_cw_to_result(json_results: dict):
    """Flatten the json results to value and columns"""
    all_results = []
    for v in json_results:
        model = v["model"]
        dataset = v["dataset"]
        for kk in ["case_f1", "case_precision", "case_recall", "lw_dice", "lw_precision", "lw_recall"]:
            for kkk in ["mean", "std"]:
                all_results.append(
                    {
                        "category": "lesion_wise_case_wise",
                        "dataset": dataset,
                        "model": model,
                        "metric": kk + "_" + kkk,
                        "value": v[kk][kkk],
                    }
                )
    return all_results


def lw_dw_to_result(json_results: dict):
    """Flatten the json results to value and columns"""
    all_results = []
    for v in json_results:
        model = v["model"]
        dataset = v["dataset"]
        for kk in ["panoptic_dice", "dataset_precision", "dataset_recall", "dataset_f1"]:
            all_results.append(
                {
                    "category": "lesion_wise_dataset_wise",
                    "dataset": dataset,
                    "model": model,
                    "metric": kk,
                    "value": v[kk],
                }
            )
        for kk in ["dice", "precision", "recall"]:
            for kkk in ["mean", "std"]:
                all_results.append(
                    {
                        "category": "lesion_wise_dataset_wise",
                        "dataset": dataset,
                        "model": model,
                        "metric": "tp_" + kk + "_" + kkk,
                        "value": v["statistics_true_positives"][kk][kkk],
                    }
                )
    return all_results


def save_datasetwise_results(sem_json: dict, instance_json: dict, datasetwise_json: dict):
    """Provide results in four columns"""
    # Move the Models to the Columns (A,B,C,D)
    sem_results = sem_json_to_result(sem_json)
    lwcw_results = lw_cw_to_result(instance_json)
    dataset_wide_results = lw_dw_to_result(datasetwise_json)
    joint_result_df = pd.DataFrame(sem_results + lwcw_results + dataset_wide_results)
    joint_result_df = joint_result_df.set_index(["category", "metric"])
    pivot_df = joint_result_df.pivot(columns=["dataset", "model"], values="value")
    pivot_df_sorted = pivot_df.sort_index(axis=1)  # Sort the column index of the models
    pivot_df_sorted.to_excel("big_table.xlsx")
    return


def test_mprage_space_caseinstance_result_table(instance_json: dict):
    """Provide results in four columns"""
    # Move the Models to the Columns (A,B,C,D)
    lwcw_results = lw_cw_to_result(instance_json)
    joint_result_df = pd.DataFrame(lwcw_results)
    joint_result_df = joint_result_df[joint_result_df["dataset"].isin(["Test Set MPRAGE", "Test Set SPACE"])]
    joint_result_df = joint_result_df[
        joint_result_df["metric"].isin(
            [
                "case_f1_mean",
                "case_f1_std",
                "case_precision_mean",
                "case_precision_std",
                "case_recall_mean",
                "case_recall_std",
            ]
        )
    ]
    joint_result_df = joint_result_df.drop(columns=["category"])
    format_mean_std = lambda mean, std: f"""{mean:.1%} ± {std:.1%}"""

    joint_result_df = joint_result_df.set_index(["model"])
    pivot_df = joint_result_df.pivot(columns=["dataset", "metric"], values="value")
    for ds in ["Test Set MPRAGE", "Test Set SPACE"]:
        for k, kk in [(("Sensitivity"), "recall"), ("Precision", "precision"), ("F1 Score", "f1")]:
            pivot_df[(ds, k)] = pivot_df.apply(
                lambda row: format_mean_std(row[(ds, f"case_{kk}_mean")], row[(ds, f"case_{kk}_std")]), axis=1
            )
            pivot_df.drop(columns=[(ds, f"case_{kk}_mean"), (ds, f"case_{kk}_std")], inplace=True)

    pivot_df_sorted = pivot_df.sort_index(axis=1)  # Sort the column index of the models
    pivot_df_sorted.to_excel("all_settings_space_mprage_dataset_results.xlsx")
    return


def save_casewise_latex_table(cwlw_df: pd.DataFrame):
    # Your existing code...
    # Convert back to flat_df
    cols_of_interest = [
        ("case_precision", "mean"),
        ("case_precision", "std"),
        ("case_recall", "mean"),
        ("case_recall", "std"),
        ("case_f1", "mean"),
        ("case_f1", "std"),
    ]
    indices = ["model", "dataset"]
    data_model_of_interest = [
        ("Test Set MPRAGE", "Setting A; v1"),
        ("Test Set MPRAGE", "Setting B; v1"),
        ("Test Set SPACE", "Setting C; v1"),
        ("Test Set SPACE", "Setting D; v1"),
    ]

    models_of_interest = cwlw_df[cols_of_interest]
    filtered_df = models_of_interest[models_of_interest.index.isin(data_model_of_interest)]
    df = filtered_df.reorder_levels(indices)

    format_mean_std = lambda mean, std: f"""{mean:.1%} ± {std:.1%}""".replace("%", r"\%")
    # Apply the formatting to create a new column
    df["PPV"] = df.apply(
        lambda row: format_mean_std(row[("case_precision", "mean")], row[("case_precision", "std")]), axis=1
    )
    df["SN"] = df.apply(
        lambda row: format_mean_std(row[("case_recall", "mean")], row[("case_recall", "std")]), axis=1
    )
    df["F1 Score"] = df.apply(lambda row: format_mean_std(row[("case_f1", "mean")], row[("case_f1", "std")]), axis=1)
    df = df.drop(
        columns=[
            ("case_precision", "mean"),
            ("case_precision", "std"),
            ("case_recall", "mean"),
            ("case_recall", "std"),
            ("case_f1", "mean"),
            ("case_f1", "std"),
        ]
    )

    # Your existing code...
    with open("table.txt", "w") as file:
        file.write(df.to_latex(index=True, escape=False))


def main():
    root_dir = Path("/mnt/cluster-data-all/t006d/nnunetv2/test_data_folder/")
    ds_pred_gt_pairs = [
        # ("test_set_MPRAGE", "pred200"),
        # ("test_set_MPRAGE", "pred201"),
        # ("test_set_MPRAGE", "pred202"),
        # ("test_set_MPRAGE", "pred203"),
        ("test_set_MPRAGE", "task561_mmm_predsTs"),
        ("test_set_MPRAGE", "task610_msm_predsTs"),
        ("test_set_MPRAGE", "task580_smm_predsTs"),
        ("test_set_MPRAGE", "task600_ssm_predsTs"),
        # ("test_set_SPACE", "pred200"),
        # ("test_set_SPACE", "pred201"),
        # ("test_set_SPACE", "pred202"),
        # ("test_set_SPACE", "pred203"),
        ("test_set_SPACE", "task590_mms_predsTs"),
        ("test_set_SPACE", "task620_sms_predsTs"),
        ("test_set_SPACE", "task570_mss_predsTs"),
        ("test_set_SPACE", "task550_sss_predsTs"),
        # ("stanford_bm", "pred200"),
        # ("stanford_bm", "pred202"),
        ("stanford_bm", "task561_mmm_15"),
        ("stanford_bm", "task570_mss"),
        # ("institutional_hd_bm", "pred200"),
        # ("institutional_hd_bm", "pred202"),
        ("institutional_hd_bm", "task561_mmm_15"),
        ("institutional_hd_bm", "task570_mss"),
        # ("bm_external", "pred200"),
        # ("bm_external", "pred202"),
        ("bm_external", "task561_mmm_15"),
        ("bm_external", "task570_mss"),
        ("brain_tr_gammaknife", "task561_mmm_15"),
        ("brain_tr_gammaknife", "task570_mss"),
        ("brainmets_molab", "task561_mmm_15"),
        ("brainmets_molab", "task570_mss"),
    ]
    sem_json = aggregate_semantic_means_results(root_dir, ds_pred_gt_pairs, 1)
    ins_json = aggregate_instance_means_results(root_dir, ds_pred_gt_pairs, 1)
    diw_json = aggregate_dataset_wide_means_results(root_dir, ds_pred_gt_pairs, 1)
    nice_sem_df = create_nice_dfs(sem_json)
    nice_inst_df = create_nice_dfs(ins_json)
    nice_lw_inst_df = create_nice_dfs(diw_json)

    save_pretty_excel(
        [nice_sem_df, nice_inst_df, nice_lw_inst_df],
        ["semantic", "lesion_wise_case_average", "lesion_wise_dataset_wide"],
    )
    # save_big_latex_table(
    #     [nice_sem_df, nice_inst_df, nice_lw_inst_df],
    #     ["semantic", "lesion_wise_case_average", "lesion_wise_dataset_wide"],
    # )
    save_casewise_latex_table(nice_inst_df)
    save_datasetwise_results(sem_json, ins_json, diw_json)
    test_mprage_space_caseinstance_result_table(ins_json)


if __name__ == "__main__":
    main()
