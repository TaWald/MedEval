import os
from pathlib import Path
import re

dataset_pattern = r"^Testdata\d\d\d\d_.*"
image_folder_pattern = "images"
label_folder_pattern = "labels"



def test_data_has_images(test_data_path: Path) -> bool:
    pass


def find_inference_directory(test_data_id: int) -> Path:
    """Finds the inference directory for a given test data id and returns the path to it.
    If the test data id is not found, raises a ValueError.
    :param test_data_id: The test data id for which to find the inference directory.
    :return: The path to the inference directory.
    """
    try:
        test_path = os.environ.get("nnunet_testing")
    except KeyError:
        print(
            "Please set the environment variable `nnunet_testing` when trying to find a test dataset through an ID."
        )
        exit(1)
    content = os.listdir(test_path)
    id_mappings = {int(re.findall(r"\d+", c)[0]): c for c in content if re.match(dataset_pattern, c)}
    if test_data_id not in id_mappings:
        raise ValueError(f"Test data with id {test_data_id} not found. Found {id_mappings}.")
    path_to_data = Path(test_path) / id_mappings[test_data_id]
    return path_to_data
