import unittest
import os
from pathlib import Path
from nneval.utils.datastructures import SemanticPair
from nneval.utils.io import get_matching_semantic_pairs

ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent / "tests" / "test_data"
SINGLE_CLASS_PD = ROOT_DIR / "single_class" / "preds"
SINGLE_CLASS_GT = ROOT_DIR / "single_class" / "gts"
MULTI_CLASS_PD = ROOT_DIR / "multi_class" / "preds"
MULTI_CLASS_GT = ROOT_DIR / "multi_class" / "gts"
SINGLE_CASE_IDs = ["case_1_5066436.nii.gz", "case_2_5452842.nii.gz", "case_3_5260428.nii.gz", "case_4_5266271.nii.gz"]
MULTI_CASE_IDs = ["case_1_6608269.nii.gz", "case_2_6740938.nii.gz", "case_3_5386003.nii.gz", "case_4_6602324.nii.gz"]


def test_find_matching_semantic_pairs_single_class():
    """Test the cases are found correctly."""
    returned_paths = get_matching_semantic_pairs(gt_path=SINGLE_CLASS_GT, pd_path=SINGLE_CLASS_PD)

    expected_semantic_pair = [
        SemanticPair(pd_p=SINGLE_CLASS_PD / cid, gt_p=SINGLE_CLASS_GT / cid) for cid in SINGLE_CASE_IDs
    ]
    assert set(returned_paths) == set(expected_semantic_pair)


def test_find_matching_semantic_pairs_multi_class():
    """Test the cases are found correctly."""
    returned_paths = get_matching_semantic_pairs(gt_path=MULTI_CLASS_GT, pd_path=MULTI_CLASS_PD)

    expected_semantic_pair = [
        SemanticPair(pd_p=MULTI_CLASS_PD / cid, gt_p=MULTI_CLASS_GT / cid) for cid in MULTI_CASE_IDs
    ]
    assert set(returned_paths) == set(expected_semantic_pair)


if __name__ == "__main__":
    unittest.main()
