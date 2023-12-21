import argparse


def get_merge_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cls",
        type=int,
        required=True,
        help="Class id for which to merge results.",
    )
    parser.add_argument(
        "-i",
        "--input_paths",
        type=str,
        required=True,
        help="Multiple paths to the root results directory.",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        help="Path to the output dir.",
        default=None,
    )
    return parser


def get_samplewise_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gt",
        "--groundtruth_path",
        type=str,
        required=True,
        help="Path to the groundtruth niftis dir.",
    )
    parser.add_argument(
        "-pd",
        "--prediction_path",
        type=str,
        required=True,
        help="Path to the prediction groundtruth dir.",
    )
    parser.add_argument(
        "-out",
        "--output_path",
        type=str,
        required=False,
        help="Path to the output dir.",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--classes_of_interest",
        type=int,
        required=False,
        help="Class of interest",
        default=[0, 1],
        nargs="+",
    )
    parser.add_argument(
        "-nproc",
        "--n_processes",
        type=int,
        required=False,
        help="Number of processes to use. If more than one does multiprocessing.",
        default=1,
    )
    return parser
