import os
from pathlib import Path


def pred_dataset(pred_path, gt_path, coi, n_processes):
    res = os.system(
        f"""python3 /home/tassilowald/Code/semantic_to_instance/sem2ins/run_evaluation.py --groundtruth_path {gt_path} --prediction_path {pred_path} --classes_of_interest {coi} --n_processes {n_processes}"""
    )
    if res != 0:
        raise ValueError("Evaluation failed")


def main():
    root_dir = Path("/mnt/cluster-data-all/t006d/nnunetv2/test_data_folder/")
    classes_of_interest = [1]

    ds_pred_gt_pairs = [
        # ("test_set_MPRAGE", "pred200", "labelsTs_Sequence_MPRAGE"),
        # ("test_set_MPRAGE", "pred201", "labelsTs_Sequence_SPACE"),
        # ("test_set_MPRAGE", "pred202", "labelsTs_Sequence_MPRAGE"),
        # ("test_set_MPRAGE", "pred203", "labelsTs_Sequence_SPACE"),
        ("test_set_MPRAGE", "task561_mmm_predsTs", "labelsTs_Sequence_MPRAGE"),
        ("test_set_MPRAGE", "task610_msm_predsTs", "labelsTs_Sequence_MPRAGE"),
        ("test_set_MPRAGE", "task580_smm_predsTs", "labelsTs_Sequence_SPACE"),
        ("test_set_MPRAGE", "task600_ssm_predsTs", "labelsTs_Sequence_SPACE"),
        # ("test_set_SPACE", "pred200", "labelsTs_Sequence_MPRAGE"),
        # ("test_set_SPACE", "pred201", "labelsTs_Sequence_SPACE"),
        # ("test_set_SPACE", "pred202", "labelsTs_Sequence_MPRAGE"),
        # ("test_set_SPACE", "pred203", "labelsTs_Sequence_SPACE"),
        ("test_set_SPACE", "task590_mms_predsTs", "labelsTs_Sequence_MPRAGE"),
        ("test_set_SPACE", "task570_mss_predsTs", "labelsTs_Sequence_MPRAGE"),
        ("test_set_SPACE", "task620_sms_predsTs", "labelsTs_Sequence_SPACE"),
        ("test_set_SPACE", "task550_sss_predsTs", "labelsTs_Sequence_SPACE"),
    ]
    # Own datasets
    for ds, pd, gt in ds_pred_gt_pairs:
        data_path = root_dir / ds
        pred_path = data_path / pd
        gt_path = data_path / gt
        for coi in classes_of_interest:
            pred_dataset(pred_path, gt_path, coi, 16)

    add_gt_pairs = [
        ("brainmets_molab", "task561_mmm_15", "labelsTr"),
        ("brainmets_molab", "task570_mss", "labelsTr"),
        ("brain_tr_gammaknife", "task561_mmm_15", "labelsTr"),
        ("brain_tr_gammaknife", "task570_mss", "labelsTr"),
        ("stanford_bm", "task561_mmm_15", "labelsTr"),
        ("stanford_bm", "task570_mss", "labelsTr"),
        ("institutional_hd_bm", "task561_mmm_15", "labelsTs"),
        ("institutional_hd_bm", "task570_mss", "labelsTs"),
        ("bm_external", "task561_mmm_15", "labelsTs"),
        ("bm_external", "task570_mss", "labelsTs"),
    ]
    # Additional datasets
    for ds, pd, gt in add_gt_pairs:
        data_path = root_dir / ds
        pred_path = data_path / pd
        gt_path = data_path / gt
        for coi in classes_of_interest:
            pred_dataset(pred_path, gt_path, coi, 16)
    # Own datasets


if __name__ == "__main__":
    main()
