from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, get_args
from loguru import logger

from nneval.evaluate_semantic import semantic_evaluation
from nneval.evaluate_instance import instance_evaluation
from nneval.generate_instances import instance_generation
from nneval.utils.io import load_yaml
from tqdm import tqdm

valid_entrypoints = Literal["semantic_evaluation", "instance_evaluation", "instance_generation"]


def convert_classes_of_interest_to_tuple(experiment_values: dict) -> dict:

    if isinstance(experiment_values["classes_of_interest"], int):
        experiment_values["classes_of_interest"] = (experiment_values["classes_of_interest"],)
    else:
        experiment_values["classes_of_interest"] = tuple(experiment_values["classes_of_interest"])
    return experiment_values


def verify_and_clean_config(config: dict):
    """Verifies the configuration file for the entrypoints and mandatory keys."""
    assert all(
        [k in get_args(valid_entrypoints) for k in config.keys()]
    ), f"Invalid entrypoint in configuration: {config.keys()} \n Choose from {valid_entrypoints}"

    k: str
    v: dict
    for k, v in config.items():
        if k == "semantic_evaluation":
            common_root_path = v.get("common_root_path", None)
            mandatory_keys = ["semantic_pd_path", "semantic_gt_path", "classes_of_interest", "output_path"]
            assert isinstance(v["runs"], list), f"Expected list for semantic_evaluation; Got {type(v)}"
            for cnt, eval in enumerate(v["runs"]):
                eval = convert_classes_of_interest_to_tuple(eval)
                if common_root_path is not None:
                    eval["semantic_pd_path"] = common_root_path / Path(eval["semantic_pd_path"])
                    eval["semantic_gt_path"] = common_root_path / Path(eval["semantic_gt_path"])
                    eval["output_path"] = common_root_path / Path(eval["output_path"])
                else:
                    eval["semantic_pd_path"] = Path(eval["semantic_pd_path"])
                    eval["semantic_gt_path"] = Path(eval["semantic_gt_path"])
                    eval["output_path"] = Path(eval["output_path"])
                assert all(
                    [kk in eval.keys() for kk in mandatory_keys]
                ), f"Missing keys in semantic_evaluation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
            del common_root_path
        elif k == "instance_evaluation":
            common_root_path = v.get("common_root_path", None)
            mandatory_keys = ["instance_pd_path", "instance_gt_path", "classes_of_interest", "output_path"]
            assert isinstance(v["runs"], list), f"Expected list for instance_evaluation; Got {type(v)}"
            for cnt, eval in enumerate(v["runs"]):
                eval = convert_classes_of_interest_to_tuple(eval)
                if common_root_path is not None:
                    eval["instance_pd_path"] = common_root_path / Path(eval["instance_pd_path"])
                    eval["instance_gt_path"] = common_root_path / Path(eval["instance_gt_path"])
                    eval["output_path"] = common_root_path / Path(eval["output_path"])
                else:
                    eval["instance_pd_path"] = Path(eval["instance_pd_path"])
                    eval["instance_gt_path"] = Path(eval["instance_gt_path"])
                    eval["output_path"] = Path(eval["output_path"])
                assert all(
                    [kk in eval.keys() for kk in mandatory_keys]
                ), f"Missing keys in instance_evaluation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
            del common_root_path
        elif k == "instance_generation":
            common_root_path = v.get("common_root_path", None)
            mandatory_keys = ["semantic_labels_path", "is_prediction"]
            assert isinstance(v["runs"], list), f"Expected list for instance_creation; Got {type(v)}"
            for cnt, eval in enumerate(v["runs"]):
                if common_root_path is not None:
                    eval["semantic_labels_path"] = common_root_path / Path(eval["semantic_labels_path"])
                    if "output_path" in eval:
                        eval["output_path"] = common_root_path / Path(eval["output_path"])
                else:
                    eval["semantic_labels_path"] = Path(eval["semantic_labels_path"])
                    if "output_path" in eval:
                        eval["output_path"] = Path(eval["output_path"])
                assert all(
                    [kk in eval.keys() for kk in mandatory_keys]
                ), f"Missing keys in instance_creation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
            del common_root_path
        else:
            raise ValueError("Invalid entrypoint in configuration")
    logger.info("Configuration verified")
    return config


def start_evaluations(config: dict):
    logger.info("Starting configurations ... ")
    for k, v in config.items():
        if k == "semantic_evaluation":
            for eval in tqdm(v["runs"], desc="Running Semantic Evaluations"):
                semantic_evaluation(**eval)
        elif k == "instance_evaluation":
            for eval in tqdm(v["runs"], desc="Running Instance Evaluations"):
                instance_evaluation(**eval)
        else:  # k == "instance_creation"
            for eval in tqdm(v["runs"], desc="Running Instance Creations"):
                instance_generation(**eval)


def start_from_config():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    config_path = args.config
    logger.info(f"Loading configuration from {config_path}")
    config = load_yaml(config_path)
    config = verify_and_clean_config(config)
    start_evaluations(config)

if __name__ == "__main__":
    start_from_config()
