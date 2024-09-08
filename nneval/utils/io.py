import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import json

import yaml
from nneval.utils.datastructures import PredGTPair

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


def save_json(data: dict, path: str | Path):
    with open(str(path), "w") as f:
        json.dump(data, f, indent=4)


def get_default_output_path(prediction_path: Path) -> Path:
    """
    Returns the default output path for the evaluation results.
    """
    out_path = prediction_path / "nneval"
    out_path.mkdir(parents=False, exist_ok=True)
    return out_path


def _get_matching_pairs(gt_path: Path | str, pd_path: Path | str, patterns: tuple[str]) -> list[PredGTPair]:
    pd_path = Path(pd_path)
    gt_path = Path(gt_path)

    gt_sample_ids = set(os.listdir(gt_path))
    gt_sample_ids = list(sorted([x for x in gt_sample_ids if x.endswith(patterns)]))

    pd_sample_ids = set(os.listdir(pd_path))
    pd_sample_ids = list(sorted([x for x in pd_sample_ids if x.endswith(patterns)]))

    pd_wo_ext = [x.split(".")[0] for x in pd_sample_ids]
    gt_wo_ext = [x.split(".")[0] for x in gt_sample_ids]
    assert set(pd_wo_ext) == set(gt_wo_ext), "PatientIDs differ from GT to PD!"

    return list([PredGTPair(pd_p=pd_path / x, gt_p=gt_path / x) for x in gt_wo_ext])


def get_matching_semantic_pairs(gt_path: Path | str, pd_path: Path | str) -> list[PredGTPair]:
    """
    Finds the matching prediction instance to groundtruth instances in the provided paths.
    If not all groundtruth instances have a matching prediction instance raises an AssertionError.
    """

    return _get_matching_pairs(gt_path=gt_path, pd_path=pd_path, patterns=SUPPORTED_EXTENSIONS)


def get_matching_instance_pairs(gt_path: Path | str, pd_path: Path | str) -> list[PredGTPair]:
    """
    Gets in.nrrd files from the paths and returns the matching pairs.
    """
    return _get_matching_pairs(gt_path=gt_path, pd_path=pd_path, patterns=("in.nrrd",))


def export_results(results: list[SemanticResult | InstanceResult], output_path: Path, output_name: str | None = None):
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
    if output_name is None:
        if isinstance(results[0], InstanceResult):
            df.to_csv(output_path / "instance_evaluation.csv")
        else:
            df.to_csv(output_path / "semantic_evaluation.csv")
    else:
        # Points to a file
        if isinstance(results[0], InstanceResult):
            df.to_csv(output_path / (output_name + "__instance_eval.csv"))
        else:
            df.to_csv(output_path / (output_name + "__semantic_eval.csv"))


def load_yaml(path: str | Path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config
