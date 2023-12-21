from multiprocessing import Pool
import os

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as nd
from skimage import morphology as morph

import sem2ins.utils.configuration as conf


def label_prediction(
    training_data: str,
    output_path: str,
    dilation_kernel: np.ndarray,
    label_kernel: np.ndarray,
    dilation_size: int,
    segmentation_id_to_merge: int,
) -> None:
    """Creates the label of the groundtruth by doing a connected components analysis (only on the CE).

    :param training_data: Path to sample sample
    :param output_path: Current path to save the Labeled Groundtruth to
    :param dilation_kernel: Structure to dilate Contrast enhanced pixels by
    :param label_kernel: Structure to calculate connected components by
    :param dilation_size: Indicator on how big the
    :param segmentation_id_to_merge: Segmentation id used to create the instances
    :return:
    """

    ######### Loading Prediction Sample #########
    res_pd = sitk.ReadImage(training_data)
    res_pd_np = sitk.GetArrayFromImage(res_pd)

    ######### Prepare Writing Paths #########
    sample_name = os.path.basename(training_data)
    ce_output_path = os.path.join(output_path, sample_name)
    os.makedirs(os.path.dirname(ce_output_path), exist_ok=True)

    if os.path.exists(ce_output_path):
        return

    ######### Only CE #########
    bin_ce_res_pd_np = res_pd_np == segmentation_id_to_merge
    if dilation_size != 0:
        ce_closed_pd = morph.binary_dilation(image=bin_ce_res_pd_np, footprint=dilation_kernel).astype(np.int32)
    else:
        ce_closed_pd = bin_ce_res_pd_np
    label_ce, _ = nd.label(input=ce_closed_pd, structure=label_kernel)
    ce_masked_label = (bin_ce_res_pd_np * label_ce).astype(np.int32)

    ######### Writing #########
    output_prediction_1 = sitk.GetImageFromArray(ce_masked_label)
    output_prediction_1.SetOrigin(res_pd.GetOrigin())
    output_prediction_1.SetDirection(res_pd.GetDirection())
    output_prediction_1.SetSpacing(res_pd.GetSpacing())

    sitk.WriteImage(output_prediction_1, ce_output_path)

    return


def label_anisotropic_groundtruth(
    sample_path: str,
    output_path: str,
    dilation_kernel_size_mm: float,
    label_kernel_size_mm: float,
    class_of_interest: int,
):
    """Does a Connected Component Analysis for anisotropic-spaced images.
    This depends on the image geometry."""
    source_sample_path = sample_path
    sample_name = os.path.basename(source_sample_path)

    target_path = os.path.join(output_path, sample_name)
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(target_path):
        print(f"Sample '{sample_name}' found; Skipping!")
        return
    read_sample: sitk.Image = sitk.ReadImage(source_sample_path)
    sample_spacing: np.ndarray = read_sample.GetSpacing()  # Is a 3x3 Array
    sample_array = sitk.GetArrayFromImage(read_sample)

    binarized_image = sample_array == class_of_interest
    # Probably the right way to go from grid structure to a Point cloud.
    #   Based off of this

    #   Algorithm description:
    #   1) Filter the point cloud by the wanted class
    #       to remove things not of our interest.  # Potentially warn for big clouds!
    #   2) Make Voxel into 8 vetices capturing the bordering nature of the objects
    #   3) Reduce the points by logical-or to make postiive,bordering negative points
    #       one positive point.
    #   4) Calculate the pointwise (e.g. euclidean) distance between all points.
    #   5) Given the upper-bound we get a adjacency graph.
    #       Depending on the computational demands for these steps it might be
    #       relevant to evluate the distance joining for best results.
    #   6) Find the connected vertices in the graph by iteratively multiplying the
    #       adjacency matrix with a single selected starting point in the matrix.
    #       Once the two matrices are not changing all points are assigned one class.
    #   7) Once all points have been assigned one class --> done.
    #   8) Use Point ID to go back to the grid positioning.
    #   9) Label the different voxels accordingly.
    #   10)


def label_groundtruth(
    path_to_sample,
    output_path: str,
    dilation_kernel: np.ndarray,
    label_kernel: np.ndarray,
    dilation_size: int,
    segmentation_id_to_merge: int,
):
    """Creates the label of the groundtruth by doing a connected components analysis (only on the CE).

    :param path_to_sample: Path to the current sample
    :param output_path: Current path to save the Labeled Groundtruth to
    :param dilation_kernel: Structure to dilate Contrast enhanced pixels by
    :param label_kernel: Structure to calculate connected components by
    :param dilation_size: Indicator on how big the
    :param segmentation_id_to_merge: Id of the segmentation that has to be merged
    :return:
    """
    source_sample_path = path_to_sample
    sample_name = os.path.basename(path_to_sample)

    ce_target_dir = os.path.join(output_path)

    os.makedirs(ce_target_dir, exist_ok=True)
    # os.makedirs(ce_ede_target_dir, exist_ok=True)

    ce_target_sample_path = os.path.join(ce_target_dir, sample_name)
    # ce_ede_target_sample_path = os.path.join(ce_ede_target_dir, sample_name)
    if os.path.exists(ce_target_sample_path):  # and os.path.exists(ce_ede_target_sample_path):
        return

    resampled_gt = sitk.ReadImage(source_sample_path)
    resampled_gt_npy: np.ndarray = sitk.GetArrayFromImage(resampled_gt)

    ######################### Only CE #########################
    bin_ce_gt_np = resampled_gt_npy == segmentation_id_to_merge
    if dilation_size != 0:
        ce_morphed_gt = morph.binary_dilation(image=bin_ce_gt_np, footprint=dilation_kernel).astype(int)
        # This step sucks ass
        # (Maybe I should replace it by doing a convolution with torch?
        # Its basically the same operation, just that it will be much faster

    else:
        ce_morphed_gt = bin_ce_gt_np

    ce_label, _ = nd.label(ce_morphed_gt, label_kernel)
    # Actually it could also be the labeling step.
    #   It might take much longer,
    #   since the dilation leads to bigger
    #   regions and therefore more pixels that are connected?
    # Profiling of this idefinelty necessary next time.

    ce_label = np.where(
        bin_ce_gt_np, ce_label, np.zeros_like(ce_label)
    )  # Make sure one does not modify the original labels

    ce_label = ce_label.astype(np.int32)
    ce_labeled_gt_img = sitk.GetImageFromArray(ce_label)
    ce_labeled_gt_img.SetSpacing(resampled_gt.GetSpacing())
    ce_labeled_gt_img.SetDirection(resampled_gt.GetDirection())
    ce_labeled_gt_img.SetOrigin(resampled_gt.GetOrigin())
    sitk.WriteImage(ce_labeled_gt_img, ce_target_sample_path)

    return None


def read_gt_pd_instances_per_image(sample_name: str) -> dict:
    ce_labeled_gt_ball_dir = os.path.join(conf.train_label_gt_location, conf.param_name_convention.format("ball", 0))
    ce_labeled_pd_ball_dir = os.path.join(conf.train_label_pd_location, conf.param_name_convention.format("ball", 0))

    ### 1. Load the images
    ### 2. Get the array
    ### 3. max the array to get the num of instances
    ### 4. return a dict mit Sample name, prediction max_ids, grondtruth max_ids

    read_image = sitk.ReadImage(os.path.join(ce_labeled_gt_ball_dir, sample_name))
    read_image_np = sitk.GetArrayFromImage(read_image)
    ce_gt_max_np = int(np.max(read_image_np))

    del read_image, read_image_np

    # read_image = sitk.ReadImage(os.path.join(ce_ede_labeled_gt_dir, sample_name))
    # read_image_np = sitk.GetArrayFromImage(read_image)
    # ce_ede_gt_max_np = int(np.max(read_image_np))
    # del read_image, read_image_np

    read_image = sitk.ReadImage(os.path.join(ce_labeled_pd_ball_dir, sample_name))
    read_image_np = sitk.GetArrayFromImage(read_image)
    ce_pd_max_np = int(np.max(read_image_np))
    del read_image, read_image_np

    # read_image = sitk.ReadImage(os.path.join(ce_ede_labeled_pd_dir, sample_name))
    # read_image_np = sitk.GetArrayFromImage(read_image)
    # ce_ede_pd_max_np = int(np.max(read_image_np))
    # del read_image, read_image_np

    result = dict(
        sample_name=sample_name,
        ce_groundtruth_instances=ce_gt_max_np,
        # ce_ede_groundtruth_instances=ce_ede_gt_max_np,
        ce_prediction_instances=ce_pd_max_np,
    )
    # ce_ede_prediction_instances=ce_ede_pd_max_np)
    return result


def run_connected_components(
    data_pair: conf.DataPair,
    sorted_samples: list[conf.PdGtPair],
    use_resampled: bool,
    gt_kernel: str = "ball",
    gt_dilation: int = 3,
    pd_kernel: str = "ball",
    pd_dilation: int = 3,
    n_processes: int = 1,
):
    """Does connected component analysis on either the resampled or the raw data."""
    if use_resampled:
        cc_pd_data_paths = [os.path.join(data_pair.resample_p_pd, pdgtpair.pd_p) for pdgtpair in sorted_samples]
        cc_gt_data_paths = [os.path.join(data_pair.resample_p_gt, pdgtpair.gt_p) for pdgtpair in sorted_samples]
    else:
        cc_pd_data_paths = [os.path.join(data_pair.pd_p, pdgtpair.pd_p) for pdgtpair in sorted_samples]
        cc_gt_data_paths = [os.path.join(data_pair.gt_p, pdgtpair.gt_p) for pdgtpair in sorted_samples]

    gt_connected_components_path = os.path.join(
        data_pair.instance_p_gt,
        conf.param_name_convention.format(gt_kernel, gt_dilation),
    )
    pd_connected_components_path = os.path.join(
        data_pair.instance_p_pd,
        conf.param_name_convention.format(pd_kernel, pd_dilation),
    )

    if pd_kernel == "ball":
        pd_joining_kernel = conf.kernel_labeling[0][1]
    elif pd_kernel == "cross":
        pd_joining_kernel = conf.kernel_labeling[1][1]
    elif pd_kernel == "rectangle":
        pd_joining_kernel = conf.kernel_labeling[2][1]
    else:
        raise ValueError("Damn. Something failed")

    if gt_kernel == "ball":
        gt_joining_kernel = conf.kernel_labeling[0][1]
    elif gt_kernel == "cross":
        gt_joining_kernel = conf.kernel_labeling[1][1]
    elif gt_kernel == "rectangle":
        gt_joining_kernel = conf.kernel_labeling[2][1]
    else:
        raise ValueError("Damn. Something failed")

    gt_dilation_kernel = morph.ball(radius=gt_dilation)
    pd_dilation_kernel = morph.ball(radius=pd_dilation)

    config_strings = conf.prediction_groundtruth_matching_template.format(
        gt_kernel,
        gt_dilation,
        pd_kernel,
        pd_dilation,
    )

    print("Creating Groundtruths intances based on dilation")
    if n_processes == 1:
        for sample_path in cc_gt_data_paths:
            label_groundtruth(
                path_to_sample=sample_path,
                output_path=gt_connected_components_path,
                dilation_kernel=gt_dilation_kernel,
                label_kernel=gt_joining_kernel,
                dilation_size=gt_dilation,
                segmentation_id_to_merge=data_pair.gt_class,
            )

        for sample_path in cc_pd_data_paths:
            label_prediction(
                training_data=sample_path,
                output_path=pd_connected_components_path,
                dilation_kernel=pd_dilation_kernel,
                label_kernel=pd_joining_kernel,
                dilation_size=pd_dilation,
                segmentation_id_to_merge=data_pair.pd_class,
            )
    else:
        with Pool(n_processes) as p:
            p.starmap(
                label_groundtruth,
                zip(
                    cc_gt_data_paths,
                    [gt_connected_components_path for _ in sorted_samples],
                    [gt_dilation_kernel for _ in sorted_samples],
                    [gt_joining_kernel for _ in sorted_samples],
                    [gt_dilation for _ in sorted_samples],
                    [data_pair.gt_class for _ in sorted_samples],
                ),
            )

        print("Finding Predictions intances.")
        with Pool(n_processes) as p:
            p.starmap(
                label_prediction,
                zip(
                    cc_pd_data_paths,
                    [pd_connected_components_path for _ in sorted_samples],
                    [pd_dilation_kernel for _ in sorted_samples],
                    [pd_joining_kernel for _ in sorted_samples],
                    [pd_dilation for _ in sorted_samples],
                    [data_pair.pd_class for _ in sorted_samples],
                ),
            )
    return pd_connected_components_path, gt_connected_components_path, config_strings
