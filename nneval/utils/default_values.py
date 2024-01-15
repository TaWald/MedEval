from typing import Dict

import numpy as np

from nneval.utils.datastructures import Instance, InstanceResult


def get_empty_prediction(sample_name: str) -> Dict:
    return dict(
        type="prediction",
        image_name=sample_name,
        prediction_size=-1.0,
        pd_instance_index=-1,
        max_dice=-1,
        dice_size_intersect_union_precision_recall_pairs=[(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)],
        max_dice_gt_index=-1.0,
    )


def no_prediction_no_groundtruth(
    case_id: str,
    semantic_class_id: int,
    volume_per_voxel: float,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    dice_threshold: float,
) -> Dict:
    """Defines behavior for a case where no groundtruth or prediction instance was found for the class_id."""
    inst_result = InstanceResult(
        dice=np.NaN,
        iou=np.NaN,
        precision=np.NaN,
        recall=np.NaN,
        intersection_voxels=np.NaN,
        union_voxels=np.NaN,
        pd_voxels=np.NaN,
        gt_voxels=np.NaN,
        pd_instance_index=np.NaN,
        gt_instance_index=np.NaN,
        match=np.NaN,
        semantic_class_id=semantic_class_id,
        case_id=case_id,
        volume_per_voxel=volume_per_voxel,
        spacing=spacing,
        dimensions=dimensions,
        dice_threshold=dice_threshold,
    )
    return inst_result


def no_groundtruth_but_prediction(
    case_id: str,
    semantic_class_id: int,
    volume_per_voxel: float,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    dice_threshold: float,
    pred: Instance,
) -> InstanceResult:
    """Calculates the predicted instance without groundtruth."""
    inst_result = InstanceResult(
        dice=0.0,
        iou=0.0,
        precision=0,  # TBD
        recall=np.NaN,
        intersection_voxels=0,
        union_voxels=int(pred.voxels),
        pd_voxels=int(pred.voxels),
        gt_voxels=np.NaN,
        pd_instance_index=int(pred.index),
        gt_instance_index=np.NaN,
        match=False,
        semantic_class_id=semantic_class_id,
        case_id=case_id,
        volume_per_voxel=volume_per_voxel,
        spacing=spacing,
        dimensions=dimensions,
        dice_threshold=dice_threshold,
    )
    return inst_result


def no_prediction_but_groundtruth(
    case_id: str,
    semantic_class_id: int,
    volume_per_voxel: float,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    dice_threshold: float,
    gt: Instance,
) -> InstanceResult:
    """Calculates the predicted instance without groundtruth."""
    inst_result = InstanceResult(
        dice=0.0,
        iou=0.0,
        precision=np.NaN,  # TBD
        recall=0,
        intersection_voxels=0,
        union_voxels=int(gt.voxels),
        pd_voxels=np.NaN,
        gt_voxels=int(gt.voxels),
        pd_instance_index=np.NaN,
        gt_instance_index=int(gt.index),
        match=False,
        semantic_class_id=semantic_class_id,
        case_id=case_id,
        spacing=spacing,
        dimensions=dimensions,
        dice_threshold=dice_threshold,
        volume_per_voxel=volume_per_voxel,
    )
    return inst_result


def groundtruth_instance_without_prediction(
    case_id: str,
    semantic_class_id: int,
    volume_per_voxel: float,
    spacing: np.ndarray,
    dimensions: np.ndarray,
    dice_threshold: float,
    pred: Instance,
) -> InstanceResult:
    """Calculates the predicted instance without groundtruth."""
    inst_result = InstanceResult(
        dice=0.0,
        iou=0.0,
        precision=0,  # TBD
        recall=np.NaN,
        intersection_voxels=0,
        union_voxels=int(pred.voxels),
        pd_voxels=int(pred.voxels),
        gt_voxels=np.NaN,
        pd_instance_index=int(pred.index),
        gt_instance_index=np.NaN,
        match=False,
        semantic_class_id=semantic_class_id,
        case_id=case_id,
        volume_per_voxel=volume_per_voxel,
        spacing=spacing,
        dimensions=dimensions,
        dice_threshold=dice_threshold,
    )
    return inst_result


def get_empty_groundtruth(sample_name: str) -> Dict:
    return dict(
        type="groundtruth",
        image_name=sample_name,
        groundtruth_size=-1.0,
        gt_instance_index=-1.0,
        max_dice=-1.0,
        dice_size_intersect_union_precision_recall_pairs=[(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)],
        max_dice_pd_index=-1.0,
    )

