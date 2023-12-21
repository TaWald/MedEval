from typing import Dict

import numpy as np


def get_empty_prediction(sample_name: str) -> Dict:
    return dict(
        type="prediction",
        image_name=sample_name,
        prediction_size=-1.0,
        prediction_index=-1,
        max_dice=-1,
        dice_size_intersect_union_precision_recall_pairs=[(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)],
        max_dice_gt_index=-1.0,
    )


def get_empty_groundtruth(sample_name: str) -> Dict:
    return dict(
        type="groundtruth",
        image_name=sample_name,
        groundtruth_size=-1.0,
        groundtruth_index=-1.0,
        max_dice=-1.0,
        dice_size_intersect_union_precision_recall_pairs=[(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)],
        max_dice_pd_index=-1.0,
    )


def get_empty_strict(sample_name: str) -> Dict:
    return dict(
        type="strict",
        image_name=sample_name,
        dice=np.NaN,
        precision=np.NaN,
        recall=np.NaN,
        intersection_voxels=np.NaN,
        union_voxels=np.NaN,
        prediction_voxels=np.NaN,
        groundtruth_voxels=np.NaN,
        prediction_index=-1,
        groundtruth_index=-1,
    )
