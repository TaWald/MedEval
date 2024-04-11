from argparse import ArgumentParser
from typing import Literal, get_args
from loguru import logger
import yaml

from nneval.evaluate_semantic import semantic_evaluation
from nneval.evaluate_instance import instance_evaluation
from nneval.generate_instances import instance_generation
from tqdm import tqdm

valid_entrypoints = Literal["semantic_evaluation", "instance_evaluation", "instance_creation"]


def verify_config(config: dict):
    """Verifies the configuration file for the entrypoints and mandatory keys."""
    assert all(
        [k in get_args(valid_entrypoints) for k in config.keys()]
    ), f"Invalid entrypoint in configuration: {config.keys()} \n Choose from {valid_entrypoints}"
    for k, v in config:
        if k == "semantic_evaluation":
            mandatory_keys = ["semantic_pd_path", "semantic_gt_path", "classes_of_interest"]
            assert isinstance(v, list), f"Expected list for semantic_evaluation; Got {type(v)}"
            for cnt, eval in enumerate(v):
                assert all(
                    [k in eval.keys() for k in mandatory_keys]
                ), f"Missing keys in semantic_evaluation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
        elif k == "instance_evaluation":
            mandatory_keys = ["instance_pd_path", "instance_gt_path", "classes_of_interest"]
            assert isinstance(v, list), f"Expected list for instance_evaluation; Got {type(v)}"
            for cnt, eval in enumerate(v):
                assert all(
                    [k in eval.keys() for k in mandatory_keys]
                ), f"Missing keys in instance_evaluation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
        else:  # k == "instance_creation"
            mandatory_keys = ["semantic_pd_path", "semantic_gt_path", "output_pd_path", "output_gt_path"]
            assert isinstance(v, list), f"Expected list for instance_creation; Got {type(v)}"
            for cnt, eval in enumerate(v):
                assert all(
                    [k in eval.keys() for k in mandatory_keys]
                ), f"Missing keys in instance_creation entry {cnt}.\n Needs: {mandatory_keys}; Got {v.keys()}"
    logger.info("Configuration verified")


def start_evaluations(config: dict):
    logger.info("Starting configurations ... ")
    for k, v in config:
        if k == "semantic_evaluation":
            for eval in tqdm(v, desc="Running Semantic Evaluations"):
                semantic_evaluation(**eval)
        elif k == "instance_evaluation":
            for eval in tqdm(v, desc="Running Instance Evaluations"):
                instance_evaluation(**eval)
        else:  # k == "instance_creation"
            for eval in tqdm(v, desc="Running Instance Creations"):
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
    config = yaml.load(config_path)
    config = verify_config(config)
    start_evaluations(config)


if __name__ == "__main__":
    start_from_config()
