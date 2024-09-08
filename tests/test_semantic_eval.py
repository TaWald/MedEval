from typing import Sequence
import unittest
import os
from pathlib import Path

import numpy as np
from nneval.utils.datastructures import PredGTPair
from monai import metrics
import SimpleITK as sitk


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent / "tests" / "test_data"
SINGLE_CLASS_PD = ROOT_DIR / "single_class" / "preds"
SINGLE_CLASS_GT = ROOT_DIR / "single_class" / "gts"
MULTI_CLASS_PD = ROOT_DIR / "multi_class" / "preds"
MULTI_CLASS_GT = ROOT_DIR / "multi_class" / "gts"
SINGLE_CASE_IDs = ["case_1_5066436.nii.gz", "case_2_5452842.nii.gz", "case_3_5260428.nii.gz", "case_4_5266271.nii.gz"]
MULTI_CASE_IDs = ["case_1_6608269.nii.gz", "case_2_6740938.nii.gz", "case_3_5386003.nii.gz", "case_4_6602324.nii.gz"]


def get_single_semantic_pair(case_id: int):
    expected_semantic_pair = [
        PredGTPair(pd_p=SINGLE_CLASS_PD / case_id, gt_p=SINGLE_CLASS_GT / case_id) for case_id in SINGLE_CASE_IDs
    ]
    return expected_semantic_pair[case_id]


def get_multi_semantic_pair(case_id: int):
    expected_semantic_pair = [
        PredGTPair(pd_p=MULTI_CLASS_PD / case_id, gt_p=MULTI_CLASS_GT / case_id) for case_id in MULTI_CASE_IDs
    ]
    return expected_semantic_pair[case_id]


def get_multiclass_compatible_pd_gt(semantic_pair: PredGTPair, class_ids_to_evaluate: Sequence[int]):
    pd_arr = sitk.GetArrayFromImage(sitk.ReadImage(semantic_pair.pd_p))
    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(semantic_pair.gt_p))

    all_pds = []
    all_gts = []
    for c_id in class_ids_to_evaluate:
        all_pds.append(pd_arr == c_id)
        all_gts.append(gt_arr == c_id)
    pds = np.stack(all_pds)[None, ...]  # Now in monai compatible format
    gts = np.stack(all_gts)[None, ...]  # Now in monai compatible format

    return pds, gts


# def test_semantic_eval_single_class():
#     """Test the cases are found correctly."""
#     class_ids_to_evaluate = (1,)
#     dice_metric = metrics.DiceMetric(include_background=False, reduction="none")

#     for i in [0, 1, 2, 3]:
#         semantic_pair = get_single_semantic_pair(i)
#         res = evaluate_semantic_results([semantic_pair], class_ids_to_evaluate)
#         multiclass_pd, multiclass_gt = get_multiclass_compatible_pd_gt(semantic_pair, class_ids_to_evaluate)
#         dsc = miseval.calc_DSC(truth=multiclass_gt, pred=multiclass_pd)
#         own_dice = res[0].dice
#         assert np.isclose(own_dice, dsc)


# def test_find_matching_semantic_pairs_multi_class():
#     """Test the cases are found correctly."""
#     returned_paths = get_matching_semantic_pairs(MULTI_CLASS_GT, MULTI_CLASS_PD)

#     expected_semantic_pair = [
#         SemanticPair(pd_p=MULTI_CLASS_PD / case_id, gt_p=MULTI_CLASS_GT / case_id) for case_id in CASE_IDs
#     ]
#     assert set(returned_paths) == set(expected_semantic_pair)


if __name__ == "__main__":
    unittest.main()
