import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import json
from nneval.utils.datastructures import InstancePair, SemanticPair

from nneval.utils.datastructures import InstanceResult, SemanticResult
import pandas as pd
from nneval.utils.naming_conventions import INSTANCE_SEG_PATTERN, SEMANTIC_SEG_PATTERN, SUPPORTED_EXTENSIONS


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


def get_all_sample_names(dir_path: str | Path) -> list[str]:
    samples = [d for d in dir_path.iterdir() if d.name.endswith(SUPPORTED_EXTENSIONS)]
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


def get_matching_semantic_pairs(gt_path: Path | str, pd_path: Path | str) -> list[SemanticPair]:
    """
    Finds the matching prediction instance to groundtruth instances in the provided paths.
    If not all groundtruth instances have a matching prediction instance raises an AssertionError.
    """
    gt_sample_ids = set(os.listdir(gt_path))
    gt_sample_ids = list(sorted([x for x in gt_sample_ids if x.endswith(SUPPORTED_EXTENSIONS)]))

    pd_sample_ids = set(os.listdir(pd_path))
    pd_sample_ids = list(sorted([x for x in pd_sample_ids if x.endswith(SUPPORTED_EXTENSIONS)]))

    pd_wo_ext = [x for x in pd_sample_ids]
    gt_wo_ext = [x for x in gt_sample_ids]
    assert set(pd_wo_ext) == set(gt_wo_ext), "PatientIDs differ from GT to PD!"

    return list([SemanticPair(pd_p=pd_path / x, gt_p=gt_path / x) for x in gt_wo_ext])


def get_matching_instance_pairs(gt_path: Path | str, pd_path: Path | str) -> list[InstancePair]:
    gt_samples = set(os.listdir(gt_path))
    pd_samples = set(os.listdir(pd_path))
    gt_samples_ids = [s.split(SEMANTIC_SEG_PATTERN)[0] for s in gt_samples if "__sem__" in s]
    gt_ext = [s.split(SEMANTIC_SEG_PATTERN)[1] for s in gt_samples if "__sem__" in s][0]
    pd_samples_ids = [s.split(SEMANTIC_SEG_PATTERN)[0] for s in pd_samples if "__sem__" in s]
    pd_ext = [s.split(SEMANTIC_SEG_PATTERN)[1] for s in pd_samples if "__sem__" in s][0]
    assert set(gt_samples_ids) == set(pd_samples_ids), "PatientIDs differ from GT to PD!"
    gt_path = Path(gt_path)  # make sure its a path
    pd_path = Path(pd_path)  # make sure its a path
    all_instance_pairs: list[InstancePair] = []
    for gt, pd in zip(sorted(gt_samples_ids), sorted(pd_samples_ids)):
        sem_gt_path = gt_path / (gt + SEMANTIC_SEG_PATTERN + gt_ext)
        sem_pd_path = pd_path / (pd + SEMANTIC_SEG_PATTERN + pd_ext)
        instance_gt_path = gt_path / (gt + INSTANCE_SEG_PATTERN + gt_ext)
        instance_pd_path = pd_path / (pd + INSTANCE_SEG_PATTERN + pd_ext)
        assert sem_gt_path.exists(), f"Semantic GT path {sem_gt_path} does not exist."
        assert sem_pd_path.exists(), f"Semantic PD path {sem_pd_path} does not exist."
        assert instance_gt_path.exists(), f"Instance GT path {instance_gt_path} does not exist."
        assert instance_pd_path.exists(), f"Instance PD path {instance_pd_path} does not exist."
        all_instance_pairs.append(
            InstancePair(
                instance_pd_p=instance_pd_path,
                instance_gt_p=instance_gt_path,
                semantic_gt_p=sem_gt_path,
                semantic_pd_p=sem_pd_path,
            )
        )
    return all_instance_pairs


def export_results(results: list[SemanticResult | InstanceResult], output_path: Path):
    """
    Export the semantic results to a comprehensive csv and some user-readable inputs.
    """
    if len(results) == 0:
        return
    # -------------------------- User-readable Dataframe ------------------------- #

    records = [s.todict() for s in results]

    # -------------------------- Comprehensive Dataframe ------------------------- #
    df = pd.DataFrame(records)
    output_path.mkdir(parents=True, exist_ok=True)
    if isinstance(results[0], InstanceResult):
        df.to_csv(output_path / "instance_evaluation.csv")
    else:
        df.to_csv(output_path / "semantic_evaluation.csv")
