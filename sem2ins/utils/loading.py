import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import sem2ins.utils.configuration as conf
from sem2ins.utils.configuration import DataPair, PdGtPair
import json


def write_dict_to_json(data: dict, filepath: str):
    """
    Writes a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be written.
        filepath (str): The path where the JSON file will be saved.
    """
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_dict_from_json(file: str, path: str) -> dict:
    """
    Reads a dictionary from a JSON file.

    Args:
        file (str): The name of the JSON file.
        path (str): The path where the JSON file is located.

    Returns:
        dict: The dictionary read from the JSON file.
    """
    file_path = f"{path}/{file}.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def get_all_sample_names():
    samples = os.listdir(conf.train_gt_location)
    return samples


def get_array_from_image(filepath: str) -> np.ndarray:
    """Reads SimpleITK image and returns it as a np.array, maintaining its shape."""
    im = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(im)


def get_spacing_from_image(filepath: str) -> np.ndarray:
    """Reads SimpleITK image and returns it as a np.array, maintaining its shape."""
    im = sitk.ReadImage(filepath)
    return im.GetSpacing()


def get_flat_array_from_image(filepath: str) -> np.ndarray:
    """Reads the simple ITK image and returns the contained numpy array as a
    flattened 1D-Array.
    """
    im_np = get_array_from_image(filepath)
    flat_im = im_np.reshape(-1)
    return flat_im


def get_matching_gt_and_preds(gt_path: Path | str, pd_path: Path | str) -> list[PdGtPair]:
    """
    Finds the matching prediction instance to groundtruth instances in the provided paths.
    If not all groundtruth instances have a matching prediction instance raises an AssertionError.
    """
    gt_sample_ids = set(os.listdir(gt_path))
    gt_sample_ids = list(sorted([x for x in gt_sample_ids if x.endswith((".nii.gz", ".nrrd"))]))
    gt_ext = ".nii.gz" if gt_sample_ids[0].endswith(".nii.gz") else ".nrrd"

    pd_sample_ids = set(os.listdir(pd_path))
    pd_sample_ids = list(sorted([x for x in pd_sample_ids if x.endswith((".nii.gz", ".nrrd"))]))
    pd_ext = ".nii.gz" if pd_sample_ids[0].endswith(".nii.gz") else ".nrrd"

    pd_wo_ext = [x.split(".")[0] for x in pd_sample_ids]
    gt_wo_ext = [x.split(".")[0] for x in gt_sample_ids]
    assert set(pd_wo_ext) == set(gt_wo_ext), "PatientIDs differ from GT to PD!"

    return list([PdGtPair(pd_p=x + pd_ext, gt_p=x + gt_ext) for x in gt_wo_ext])


def get_matching_gt_and_preds_from_data_pair(datapair: DataPair) -> list[PdGtPair]:
    """
    Finds the matching prediction instance to groundtruth instances in the provided paths.
    If not all groundtruth instances have a matching prediction instance raises an AssertionError.
    """
    gt_p = datapair.gt_p
    pd_p = datapair.pd_p
    return get_matching_gt_and_preds(gt_p, pd_p)
