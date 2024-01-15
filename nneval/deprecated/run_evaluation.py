import os
from nneval.deprecated.evaluation.hard_evaluation import hard_evaluate
from nneval.deprecated.samplewise.run_samplewise import run_samplewise

from nneval.utils.parser import get_samplewise_eval_parser
from nneval.utils.io import get_matching_gt_and_preds_from_data_pair


import nneval.utils.configuration as configuration
from nneval.instancewise.connected_components import run_connected_components
from nneval.deprecated.evaluation.match_segmentations import run_tp_fp_calculation
from nneval.instancewise.resample import run_resample


def evaluate_configuration(
    data_pair: configuration.DatasetEvalInfo,
    gt_kernel: str = "ball",
    gt_dilation: int = 3,
    pd_kernel: str = "ball",
    pd_dilation: int = 3,
    n_processes: int = 1,
    do_resampling: bool = False,
):
    sorted_samples = get_matching_gt_and_preds_from_data_pair(data_pair)
    # ------------------- Samplewise evaluation  (without resampling)-------------------
    run_samplewise(data_pair, n_processes=n_processes)
    # ------------------- Resampling     -------------------
    if do_resampling:
        run_resample(data_pair, n_processes=n_processes)

    # ------------------- Connected component analysis -------------------
    print("Starting component component analysis - labeling single instances")
    pd_cc_path, gt_cc_path, config_strings = run_connected_components(
        data_pair=data_pair,
        sorted_samples=sorted_samples,
        use_resampled=do_resampling,
        gt_kernel=gt_kernel,
        gt_dilation=gt_dilation,
        pd_kernel=pd_kernel,
        pd_dilation=pd_dilation,
        n_processes=n_processes,
    )

    # ------------------- Matching groundtruth predictions   -------------------
    print("Starting matching of groundtruth and prediction instances")
    source_labeled_pd = [os.path.join(pd_cc_path, sample.pd_p) for sample in sorted_samples]
    source_labeled_gt = [os.path.join(gt_cc_path, sample.gt_p) for sample in sorted_samples]

    run_tp_fp_calculation(
        pd_labeled=source_labeled_pd,
        gt_labeled=source_labeled_gt,
        dice_threshold=0.1,
        config_strings=config_strings,
        n_processes=n_processes,
        data_pair=data_pair,
    )

    # ------------------- Create the csv files of the stuff.   -------------------
    print("Creating instance-wise evaluation of the samples")
    # soft_evaluate(
    #     config_strings,
    #     data_pair.instancewise_result_p,
    #     dice_threshold=0.1,
    #     samples=sorted_samples,
    #     filtering_size=data_pair.filtering_size,
    # )
    hard_evaluate(
        config_strings,
        data_pair.instancewise_result_p,
        samples=sorted_samples,
        filtering_size=data_pair.filtering_size,
    )
    return


def main():
    """
    In order to evaluate a prediction/groundtruth the data has to be in nnUNet format.
    (A joint folder with "labelsTr" and "preds" has to exist.)

    1. (optional) Resample the prediction to the spacing of the groundtruth
    2. Calculate the normal dice scores for the different classes per image
        2.1. Save them in a csv in a per-case basis
    3. Calculate instance based segmentation metrics via morphological operations
        3.1 Create Labeled instances:
            3.1.1. Do expansion/dilation with diff. morphological operators
            3.1.2. Do connected component analysis
        3.2 Use Labeled instances to calculate instance wise metrics
        3.3 save them in multiple csvs
    4. Visualize the different things.

    :return:
    """

    parser = get_samplewise_eval_parser()
    args = parser.parse_args()
    gt_path = args.groundtruth_path
    pd_path = args.prediction_path
    out_path = args.output_path
    cls_of_interest = args.classes_of_interest
    nproc = args.n_processes
    resampling = False
    # Currently no resampling

    if out_path is None:
        out_path = pd_path
    configs = [
        configuration.DatasetEvalInfo(
            out_p=out_path,
            gt_class=cls,
            pd_class=cls,
            gt_p=gt_path,
            pd_p=pd_path,
            filtering_size=None,
        )
        for cls in cls_of_interest
    ]

    for data_pair in configs:
        print(f"Starting to evaluate into: {data_pair.out_p}")
        evaluate_configuration(
            data_pair=data_pair,
            gt_kernel="ball",
            gt_dilation=3,
            pd_kernel="ball",
            pd_dilation=3,
            n_processes=nproc,
            do_resampling=resampling,
        )


if __name__ == "__main__":
    main()
