from multiprocessing import Pool
import os
import sys

import SimpleITK as sitk
import numpy as np
import pandas as pd
from sem2ins.utils.configuration import DataPair, PdGtPair
from sem2ins.utils.loading import (
    get_array_from_image,
    get_matching_gt_and_preds,
    get_matching_gt_and_preds_from_data_pair,
)
from sem2ins.utils.parser import get_samplewise_eval_parser


def samplewise_eval(
    gt_array: np.ndarray,
    gt_class_of_interest: int,
    pd_array: np.ndarray,
    pd_class_of_interest: int,
    sample_name: str,
    vol_per_voxel: float = 1.0,
) -> dict:
    gt_flat = np.reshape(gt_array, newshape=[-1])
    pd_flat = np.reshape(pd_array, newshape=[-1])

    gt_bool_npy = gt_flat == gt_class_of_interest
    pd_bool_npy = pd_flat == pd_class_of_interest

    #### Has to remove the instances where the prediction instance volume is smaller
    # than the threshold (only noticeable for bigger filter thresholds)

    joined_npy = np.logical_or(gt_bool_npy, pd_bool_npy)  # Where either of them is not background!

    gt_bool_npy = gt_bool_npy[joined_npy]
    pd_bool_npy = pd_bool_npy[joined_npy]

    pd_voxels = np.count_nonzero(pd_bool_npy)
    gt_voxels = np.count_nonzero(gt_bool_npy)

    union_voxels = np.count_nonzero(joined_npy)
    intersection_voxels = np.count_nonzero(np.logical_and(gt_bool_npy, pd_bool_npy))
    if union_voxels != 0:
        dice = (2 * intersection_voxels) / (union_voxels + intersection_voxels)
        iou = intersection_voxels / union_voxels
    else:
        dice = np.NAN
        iou = np.NAN

    if gt_voxels != 0:
        sensitivity = intersection_voxels / gt_voxels
    else:
        sensitivity = np.NAN

    if pd_voxels != 0:
        precision = intersection_voxels / pd_voxels
    else:
        precision = np.NAN

    return dict(
        groundtruth_voxels=gt_voxels,
        prediction_voxels=pd_voxels,
        volume_per_voxel_mm3=vol_per_voxel,
        intersection_voxels=intersection_voxels,
        union_voxels=union_voxels,
        sensitivity=sensitivity,
        precision=precision,
        dice=dice,
        iou=iou,
        pd_segmentation_class=pd_class_of_interest,
        gt_segmentation_class=gt_class_of_interest,
        gt_volume_mm3=gt_voxels * vol_per_voxel,
        pd_volume_mm3=pd_voxels * vol_per_voxel,
        sample_name=sample_name,
    )


def get_samplewise_statistics(
    samplewise_results: list[dict],
) -> dict:
    return (
        {
            "dice": {
                "mean": float(np.nanmean([entry["dice"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["dice"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["dice"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["dice"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["dice"] for entry in samplewise_results], 0.75)),
            },
            "iou": {
                "mean": float(np.nanmean([entry["iou"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["iou"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["iou"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["iou"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["iou"] for entry in samplewise_results], 0.75)),
            },
            "gt_volume_mm3": {
                "mean": float(np.nanmean([entry["gt_volume_mm3"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["gt_volume_mm3"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["gt_volume_mm3"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["gt_volume_mm3"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["gt_volume_mm3"] for entry in samplewise_results], 0.75)),
            },
            "pd_volume_mm3": {
                "mean": float(np.nanmean([entry["pd_volume_mm3"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["pd_volume_mm3"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["pd_volume_mm3"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["pd_volume_mm3"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["pd_volume_mm3"] for entry in samplewise_results], 0.75)),
            },
            "sensitivity": {
                "mean": float(np.nanmean([entry["sensitivity"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["sensitivity"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["sensitivity"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["sensitivity"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["sensitivity"] for entry in samplewise_results], 0.75)),
            },
            "precision": {
                "mean": float(np.nanmean([entry["precision"] for entry in samplewise_results])),
                "std": float(np.nanstd([entry["precision"] for entry in samplewise_results])),
                "q1": float(np.nanquantile([entry["precision"] for entry in samplewise_results], 0.25)),
                "q2": float(np.nanquantile([entry["precision"] for entry in samplewise_results], 0.5)),
                "q3": float(np.nanquantile([entry["precision"] for entry in samplewise_results], 0.75)),
            },
        },
    )


def calculate_samplewise_results(data_pair: DataPair, n_processes: int = 1) -> list[dict]:
    print("Startig unresampled casewise evaluation. (not resampled)")
    sorted_samples: list[PdGtPair] = get_matching_gt_and_preds_from_data_pair(data_pair)
    if n_processes == 1:
        samplewise_results = []
        for pdgtpair in sorted_samples:
            pd_sample = os.path.join(data_pair.pd_p, pdgtpair.pd_p)
            gt_sample = os.path.join(data_pair.gt_p, pdgtpair.gt_p)
            samplewise_results.append(
                samplewise_result(
                    gt_sample_path=gt_sample,
                    pd_sample_path=pd_sample,
                    gt_class_of_interest=data_pair.gt_class,
                    pd_class_of_interest=data_pair.pd_class,
                )
            )
    else:
        gt_samples = [os.path.join(data_pair.gt_p, pdgtpair.gt_p) for pdgtpair in sorted_samples]
        pd_samples = [os.path.join(data_pair.pd_p, pdgtpair.pd_p) for pdgtpair in sorted_samples]
        with Pool(n_processes) as p:
            samplewise_results = p.starmap(
                samplewise_result,
                zip(
                    gt_samples,
                    pd_samples,
                    [data_pair.gt_class for _ in sorted_samples],
                    [data_pair.pd_class for _ in sorted_samples],
                ),
            )
    return samplewise_results


def samplewise_result(
    gt_sample_path: str,
    pd_sample_path: str,
    gt_class_of_interest: int,
    pd_class_of_interest: int,
) -> dict:
    """Calculates the dice of the niftis given in gt_sample_path and pd_sample_path.
    The dice is calcualted between the voxels that are of gt_class_of_interest and
    pd_class_of_interest.

    :param gt_class_of_interest: Class used to compare in groundtruth
    :param pd_class_of_interest: Class used to compare in predictions
    :param gt_sample_path: Path to the gt sample to calc dice on
    :param pd_sample_path: Path to the pd sample to calc dice on
    """

    sample_name = os.path.basename(gt_sample_path)

    gt_im = sitk.ReadImage(gt_sample_path)
    pd_im = sitk.ReadImage(pd_sample_path)

    gt_spacing = gt_im.GetSpacing()
    pd_spacing = pd_im.GetSpacing()

    assert np.all(np.isclose(gt_spacing, pd_spacing)), "Not equal spacing"

    vol_factor = float(np.prod(gt_spacing))  # in mm³

    gt_npy = get_array_from_image(gt_sample_path)
    pd_npy = get_array_from_image(pd_sample_path)

    return samplewise_eval(
        gt_array=gt_npy,
        gt_class_of_interest=gt_class_of_interest,
        pd_array=pd_npy,
        pd_class_of_interest=pd_class_of_interest,
        sample_name=sample_name,
        vol_per_voxel=vol_factor,
    )


if __name__ == "__main__":
    parser = get_samplewise_eval_parser()
    args = parser.parse_args()
    gt_path = args.groundtruth_path
    pd_path = args.prediction_path
    out_path = args.output_path
    cls_of_interest = args.classes_of_interest
    if out_path is None:
        out_path = pd_path

    results = []
    sorted_ids = get_matching_gt_and_preds(gt_path, pd_path)
    for patient in sorted_ids:
        for c in cls_of_interest:
            patient_result = {"patient_id": patient, "class_id": c}
            pat_gt_path = os.path.join(gt_path, patient)
            pat_pd_path = os.path.join(pd_path, patient)
            patient_result.update(
                **samplewise_result(
                    gt_sample_path=pat_gt_path,
                    pd_sample_path=pat_pd_path,
                    gt_class_of_interest=c,
                    pd_class_of_interest=c,
                )
            )
            results.append(patient_result)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_path, "samplewise_evaluation.csv"), index=False)
    sys.exit(0)
