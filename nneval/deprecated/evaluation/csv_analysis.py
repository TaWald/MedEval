import numpy as np




def get_patient_ids_of_all_samples(groundtruth_results: list[dict]) -> set[str]:
    image_set = set()
    for groundtruth in groundtruth_results:
        image_id = groundtruth["image_name"]
        if image_id in image_set:
            continue
        else:
            image_set.add(image_id)
    return image_set


def groundtruth_metastases_per_patient(groundtruth_results: list[dict], samples: list[str]) -> list[dict]:
    """count the metastases of a patient"""
    image_names = samples
    result = []

    for image_name in image_names:
        n_metastases = len([True for groundtruth in groundtruth_results if groundtruth["image_name"] == image_name])
        result.append(dict(image_name=image_name, n_metastases=n_metastases))
    return result


def case_wise_summary_json(case_wise_res: list[dict]):
    all_precisions = []
    all_recalls = []
    all_f1 = []
    for res in case_wise_res:
        all_precisions.append(res["precision"])
        all_recalls.append(res["recall"])
        all_f1.append(res["f1_score"])

    result = {
        "mean_precision": np.nanmean(all_precisions),
        "CI_precision": confidence_intervals(all_precisions),
        "1st_quart_precision": np.nanquantile(all_precisions, 0.25),
        "2st_quart_precision": np.nanquantile(all_precisions, 0.5),
        "3rd_quart_precision": np.nanquantile(all_precisions, 0.75),
        "mean_recall": np.nanmean(all_recalls),
        "CI_recall": confidence_intervals(all_recalls),
        "1st_quart_recall": np.nanquantile(all_recalls, 0.25),
        "2st_quart_recall": np.nanquantile(all_recalls, 0.5),
        "3rd_quart_recall": np.nanquantile(all_recalls, 0.75),
        "mean_f1": np.nanmean(all_f1),
        "CI_f1": confidence_intervals(all_f1),
        "1st_quart_f1": np.nanquantile(all_f1, 0.25),
        "2st_quart_f1": np.nanquantile(all_f1, 0.5),
        "3rd_quart_f1": np.nanquantile(all_f1, 0.75),
    }
    return result


def casewise_results(
    groundtruth_results: list[dict],
    prediction_results: list[dict],
    samples: list[str],
    filter_size: int = 0,
    dice_threshold: float = 0.1,
) -> list[dict]:
    result = []
    for sample in samples:
        groundtruths_of_sample = [
            groundtruth for groundtruth in groundtruth_results if (groundtruth["image_name"] == sample)
        ]
        predictions_of_sample = [
            prediction
            for prediction in prediction_results
            if (prediction["image_name"] == sample) and (prediction["prediction_size"] > filter_size)
        ]

        n_predicted_positives = float(len(predictions_of_sample))
        n_positives = float(len(groundtruths_of_sample))

        if (n_positives == 0.0) and (n_predicted_positives == 0.0):
            result.append(
                dict(
                    true_positives=0.0,
                    false_positives=0.0,
                    false_negatives=0.0,
                    contained_groundtruth_instances=n_positives,
                    positives=n_positives,
                    predicted_positive=n_predicted_positives,
                    predicted_remaining_instances=n_predicted_positives,
                    precision=np.nan,
                    recall=np.nan,
                    f1_score=np.nan,
                    mean_pd_dice=np.nan,
                    mean_pd_precision=np.nan,
                    mean_pd_recall=np.nan,
                    mean_gt_dice=np.nan,
                    mean_gt_precision=np.nan,
                    mean_gt_recall=np.nan,
                    case_name=sample,
                ),
            )
        elif n_positives == 0 and (n_predicted_positives != 0):
            result.append(
                dict(
                    true_positives=0.0,
                    false_positives=n_predicted_positives,
                    false_negatives=0.0,
                    contained_groundtruth_instances=n_positives,
                    positives=n_positives,
                    predicted_positive=n_predicted_positives,
                    predicted_remaining_instances=n_predicted_positives,
                    precision=0.0,
                    recall=np.nan,
                    f1_score=0.0,
                    mean_pd_dice=0.0,
                    mean_pd_precision=0.0,
                    mean_pd_recall=np.nan,
                    mean_gt_dice=0.0,
                    mean_gt_precision=0.0,
                    mean_gt_recall=np.nan,
                    case_name=sample,
                )
            )
        elif n_positives != 0 and (n_predicted_positives == 0):
            result.append(
                dict(
                    true_positives=0.0,
                    false_positives=0.0,
                    false_negatives=n_positives,
                    contained_groundtruth_instances=n_positives,
                    positives=n_positives,
                    predicted_positive=n_predicted_positives,
                    predicted_remaining_instances=0.0,
                    precision=np.nan,
                    recall=0.0,
                    f1_score=0.0,  # 0.0 <--- This might be questionable?
                    mean_gt_dice=0.0,
                    mean_gt_precision=np.nan,
                    mean_gt_recall=0.0,
                    mean_pd_dice=0.0,
                    mean_pd_precision=np.nan,
                    mean_pd_recall=0.0,
                    case_name=sample,
                )
            )
        else:
            n_true_groundtruths = 0.0
            gt_max_dices = []
            gt_max_dice_precisions = []
            gt_max_dice_recalls = []
            for groundtruth in groundtruths_of_sample:
                dice_size_pairs = groundtruth["dice_size_intersect_union_precision_recall_pairs"]
                gt_dices = [
                    dice for dice, size, _, _, _, _ in dice_size_pairs if size > filter_size
                ]  # removed the predictions that do not follow the size criterium
                gt_precisions = [
                    prec for _, size, _, _, prec, recall in dice_size_pairs if size > filter_size
                ]  # removed the predictions that do not follow the size criterium
                gt_recalls = [
                    recall for _, size, _, _, prec, recall in dice_size_pairs if size > filter_size
                ]  # removed the predictions that do not follow the size criterium
                max_dice_index = np.argmax(gt_dices)
                gt_max_dices.append(gt_dices[max_dice_index])
                gt_max_dice_precisions.append(gt_precisions[max_dice_index])
                gt_max_dice_recalls.append(gt_recalls[max_dice_index])

                if (len(gt_dices) > 0) and (np.max(gt_dices) > dice_threshold):
                    n_true_groundtruths += 1

            mean_gt_dice = float(np.mean(gt_max_dices))
            mean_gt_prec = float(np.mean(gt_max_dice_precisions))
            mean_gt_rec = float(np.mean(gt_max_dice_recalls))

            false_negatives = n_positives - n_true_groundtruths

            pred_dices = [pred["max_dice"] for pred in predictions_of_sample]
            pred_precisions = [pred["max_dice_precision"] for pred in predictions_of_sample]
            pred_recalls = [pred["max_dice_recall"] for pred in predictions_of_sample]

            mean_pd_dices = np.mean(pred_dices)
            mean_pd_precisions = np.mean(pred_precisions)
            mean_pd_recalls = np.mean(pred_recalls)

            n_true_positives = len([True for pred in predictions_of_sample if pred["max_dice"] > dice_threshold])
            n_false_positives = n_predicted_positives - n_true_positives
            precision = n_true_positives / (n_true_positives + n_false_positives)
            recall = (n_positives - false_negatives) / len(groundtruths_of_sample)
            if (precision + recall) != 0:
                f1_score = (2 * precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            result.append(
                dict(
                    true_positives=float(n_true_positives),
                    false_positives=float(n_false_positives),
                    false_negatives=float(false_negatives),
                    positives=float(n_positives),
                    predicted_positive=float(n_predicted_positives),
                    contained_groundtruth_instances=float(n_positives),
                    predicted_remaining_instances=float(n_predicted_positives),
                    precision=float(precision),
                    recall=float(recall),
                    f1_score=float(f1_score),
                    case_name=sample,
                    # Unthresholded
                    mean_pd_dice=float(mean_pd_dices),
                    mean_pd_precision=float(mean_pd_precisions),
                    mean_pd_recall=float(mean_pd_recalls),
                    mean_gt_dice=float(mean_gt_dice),
                    mean_gt_precision=float(mean_gt_prec),
                    mean_gt_recall=float(mean_gt_rec),
                )
            )
    return result


def casewise_stats(casewise_result: list[dict], filter_size: int) -> dict:
    precision = [res["precision"] for res in casewise_result]
    recall = [res["recall"] for res in casewise_result]
    f1_score = [res["f1_score"] for res in casewise_result]

    n_cases = len(casewise_result)
    n_false_positives = sum([res["false_positives"] for res in casewise_result])
    n_false_negatives = sum([res["false_negatives"] for res in casewise_result])
    n_true_positives = sum([res["true_positives"] for res in casewise_result])
    n_positives = sum([res["positives"] for res in casewise_result])
    n_predicted_positives = sum([res["predicted_positive"] for res in casewise_result])

    casewise_mean_gt_dices = [res["mean_gt_dice"] for res in casewise_result]
    casewise_mean_gt_prec = [res["mean_gt_precision"] for res in casewise_result]
    casewise_mean_gt_rec = [res["mean_gt_recall"] for res in casewise_result]
    casewise_mean_pd_dices = [res["mean_pd_dice"] for res in casewise_result]
    casewise_mean_pd_prec = [res["mean_pd_precision"] for res in casewise_result]
    casewise_mean_pd_rec = [res["mean_pd_recall"] for res in casewise_result]

    result = dict(
        filter_size=filter_size,
        mean_precision=np.nanmean(precision),
        # ci95_precision=confidence_interval_binomial(p=np.nanmean(precision), n=len(precision), confidence_level=0.95),
        std_precision=np.nanstd(precision),
        median_precision=np.nanmedian(precision),
        # Recall
        mean_recall=np.nanmean(recall),
        std_recall=np.nanstd(recall),
        median_recall=np.nanmedian(recall),
        # F1 Score
        mean_f1_score=np.nanmean(f1_score),
        std_f1_score=np.nanstd(f1_score),
        median_f1_score=np.nanmedian(f1_score),
        q25_f1_score=np.nanquantile(f1_score, 0.25),
        q75_f1_score=np.nanquantile(f1_score, 0.75),
        # Groundtruth stuff
        mean_gt_dice=np.nanmean(casewise_mean_gt_dices),
        std_gt_dice=np.nanstd(casewise_mean_gt_dices),
        median_gt_dice=np.nanmedian(casewise_mean_gt_dices),
        q25_gt_dice=np.nanquantile(casewise_mean_gt_dices, 0.25),
        q75_gt_dice=np.nanquantile(casewise_mean_gt_dices, 0.75),
        mean_gt_precision=np.nanmean(casewise_mean_gt_prec),
        std_gt_precision=np.nanstd(casewise_mean_gt_prec),
        q25_gt_precision=np.nanquantile(casewise_mean_gt_prec, 0.25),
        q50_gt_precision=np.nanquantile(casewise_mean_gt_prec, 0.5),
        q75_gt_precision=np.nanquantile(casewise_mean_gt_prec, 0.75),
        mean_gt_recall=np.nanmean(casewise_mean_gt_rec),
        std_gt_recall=np.nanstd(casewise_mean_gt_rec),
        q25_gt_recall=np.nanquantile(casewise_mean_gt_rec, 0.25),
        median_gt_recall=np.nanquantile(casewise_mean_gt_rec, 0.5),
        q75_gt_recall=np.nanquantile(casewise_mean_gt_rec, 0.75),
        # Prediction stuff
        mean_pd_dice=np.nanmean(casewise_mean_pd_dices),
        std_pd_dice=np.nanstd(casewise_mean_pd_dices),
        median_pd_dice=np.nanmedian(casewise_mean_pd_dices),
        q25_pd_dice=np.nanquantile(casewise_mean_pd_dices, 0.25),
        q75_pd_dice=np.nanquantile(casewise_mean_pd_dices, 0.75),
        mean_pd_precision=np.nanmean(casewise_mean_pd_prec),
        std_pd_precision=np.nanstd(casewise_mean_pd_prec),
        q25_pd_precision=np.nanquantile(casewise_mean_pd_prec, 0.25),
        median_pd_precision=np.nanquantile(casewise_mean_pd_prec, 0.5),
        q75_pd_precision=np.nanquantile(casewise_mean_pd_prec, 0.75),
        mean_pd_recall=np.nanmean(casewise_mean_pd_rec),
        std_pd_recall=np.nanstd(casewise_mean_pd_rec),
        q25_pd_recall=np.nanquantile(casewise_mean_pd_rec, 0.25),
        median_pd_recall=np.nanquantile(casewise_mean_pd_rec, 0.5),
        q75_pd_recall=np.nanquantile(casewise_mean_pd_rec, 0.75),
        n_cases=n_cases,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        n_false_negatives=n_false_negatives,
        n_positives=n_positives,
        n_predicted_positives=n_predicted_positives,
    )
    return result


def groundtruth_volume_prediction_volume_pairs(
    groundtruth_results: list[dict],
    samples: list[str],
    size_filter_val: int = 0,
    dice_threshold=0.1,
) -> list[dict]:
    """Calculates pairs of volume of prediction and groundtruth, with corresponding Dice.

    :param groundtruth_results:
    :return:
    """
    result = []
    for sample in samples:
        groundtruth_of_sample = [
            groundtruth for groundtruth in groundtruth_results if groundtruth["image_name"] == sample
        ]
        if len(groundtruth_of_sample) == 0:
            result.append(
                dict(
                    groundtruth_size=0,
                    prediction_size=0,
                    dice_of_max_prediction=-1,
                    dice_when_connecting_multiple_predictions_of_same_groundtruth=-1,
                    prediction_size_when_connecting_multiple_predictions_of_same_groundtruth=0,
                    prediction_size_when_connecting_multiple_predictions_of_same_groundtruth_rounded=0,
                    n_joined_predictions=0,
                    n_true_positive_predictions=0,
                    n_false_positive_predictions=0,
                    image_name=sample,
                )
            )
        else:
            for groundtruth in groundtruth_of_sample:
                current_size = groundtruth["groundtruth_size"]
                dice_size_pairs = groundtruth[
                    "dice_size_intersect_union_precision_recall_pairs"
                ]  # List of Tuples [dice_val, groundtruth_size]

                remaining_pairs = [
                    (dice, size, n_in, n_un, prec, rec)
                    for dice, size, n_in, n_un, prec, rec in dice_size_pairs
                    if (size > size_filter_val) and dice != 0
                ]
                if len(remaining_pairs) == 0:
                    size_of_max_dice_prediction = 0
                    dice_of_max_prediction = 0
                    joined_dice = 0
                    total_joined_prediction_size = 0
                    n_joined_predictions = 0
                    true_positive_pds = 0
                    false_positive_pds = 0
                elif len(remaining_pairs) == 1:
                    dice = [vals[0] for vals in remaining_pairs]
                    max_id = np.argmax(dice)
                    dice_of_max_prediction = remaining_pairs[max_id][0]
                    size_of_max_dice_prediction = remaining_pairs[max_id][1]
                    joined_dice = dice
                    total_joined_prediction_size = size_of_max_dice_prediction
                    n_joined_predictions = len(remaining_pairs)
                    if dice_of_max_prediction >= dice_threshold:
                        true_positive_pds = 1
                        false_positive_pds = 0
                    else:
                        true_positive_pds = 0
                        false_positive_pds = 1

                else:  # When multiple predictions match!
                    dice = [vals[0] for vals in remaining_pairs]
                    max_id = np.argmax(dice)
                    dice_of_max_prediction = remaining_pairs[max_id][0]
                    size_of_max_dice_prediction = remaining_pairs[max_id][1]

                    #  Here multiple Predictions overlap with the groundtruth, therefore we can also just interpret them as "one instance"

                    # This actually does not hold for dice.
                    overlapping_pixels = [
                        (dice * (size + current_size)) / 2 for dice, size, _, _, _, _ in remaining_pairs
                    ]

                    # all intersections are disjoint and all predictions are disjoint, so the number of total overlapping pixels is all intersections added
                    total_intersecting_voxels = sum(overlapping_pixels)
                    total_joined_prediction_size = sum([size for dice, size, _, _, _, _ in remaining_pairs])
                    joined_dice = (2 * total_intersecting_voxels) / (current_size + total_joined_prediction_size)
                    n_joined_predictions = len(remaining_pairs)
                    true_positive_pds = len(
                        [True for dice, size, _, _, _, _ in remaining_pairs if dice >= dice_threshold]
                    )
                    false_positive_pds = n_joined_predictions - true_positive_pds

                result.append(
                    dict(
                        groundtruth_size=current_size,
                        prediction_size=size_of_max_dice_prediction,
                        dice_of_max_prediction=dice_of_max_prediction,
                        dice_when_connecting_multiple_predictions_of_same_groundtruth=joined_dice,
                        prediction_size_when_connecting_multiple_predictions_of_same_groundtruth=total_joined_prediction_size,
                        prediction_size_when_connecting_multiple_predictions_of_same_groundtruth_rounded=round(
                            total_joined_prediction_size
                        ),
                        n_joined_predictions=n_joined_predictions,
                        n_true_positive_predictions=true_positive_pds,
                        n_false_positive_predictions=false_positive_pds,
                        image_name=sample,
                    )
                )

    return result


def size_wise_matching(
    groundtruth_results: list[dict], prediction_results: list[dict], samples: list[str]
) -> list[dict]:
    """Calculates true positives, false positives, false negatives, and bunch of different shit.
    Returns a dict with the keys. To be saved in a csv file.

    :param groundtruth_results: List with dictionary for each groundtruth instance
    :param prediction_results: List with dictionary for each prediction instance
    :param samples: List of sample names
    :return:
    """
    # Prediction maximal filter size in mm³
    n_total_predictions = len(prediction_results)
    n_total_groundtruths = len(groundtruth_results)

    true_positive_dice_threshold = 0.1

    # Size wise filtering (for the/

    image_set = samples
    n_samples = len(image_set)

    size_wise_matching_results = []
    for i in range(200):
        remaining_predictions = [prediction for prediction in prediction_results if prediction["prediction_size"] > i]
        n_remaining_predictions = len(remaining_predictions)
        removed_predictions = n_total_predictions - len(remaining_predictions)

        true_positives = len(
            [
                prediction
                for prediction in remaining_predictions
                if prediction["max_dice"] >= true_positive_dice_threshold
            ]
        )
        false_positives = len(remaining_predictions) - true_positives
        assert true_positives + false_positives + removed_predictions == n_total_predictions, "Something is wrong!"

        prediction_wise_dice = [remaining_prediction["max_dice"] for remaining_prediction in remaining_predictions]
        prediction_wise_prec = [
            remaining_prediction["max_dice_precision"] for remaining_prediction in remaining_predictions
        ]
        prediction_wise_rec = [
            remaining_prediction["max_dice_recall"] for remaining_prediction in remaining_predictions
        ]

        ## Prediction stuff
        # Dice
        mean_prediction_dice = float(np.mean(prediction_wise_dice))
        std_prediction_dice = float(np.std(prediction_wise_dice))
        if len(prediction_wise_dice) == 0:
            quant25_prediction_dice = 0.0
            quant50_prediction_dice = 0.0
            quant75_prediction_dice = 0.0
        else:
            quant25_prediction_dice = float(np.quantile(prediction_wise_dice, q=0.25))
            quant50_prediction_dice = float(np.quantile(prediction_wise_dice, q=0.5))
            quant75_prediction_dice = float(np.quantile(prediction_wise_dice, q=0.75))

        # Precision
        mean_prediction_wise_precision = float(np.mean(prediction_wise_prec))
        std_prediction_wise_precision = float(np.std(prediction_wise_prec))
        if len(prediction_wise_prec) == 0:
            quant25_prediction_wise_precision = 0.0
            quant50_prediction_wise_precision = 0.0
            quant75_prediction_wise_precision = 0.0
        else:
            quant25_prediction_wise_precision = float(np.quantile(prediction_wise_prec, q=0.25))
            quant50_prediction_wise_precision = float(np.quantile(prediction_wise_prec, q=0.5))
            quant75_prediction_wise_precision = float(np.quantile(prediction_wise_prec, q=0.75))

        # Recall
        mean_prediction_wise_recall = float(np.mean(prediction_wise_rec))
        std_prediction_wise_recall = float(np.std(prediction_wise_rec))
        if len(prediction_wise_rec) == 0:
            quant25_prediction_wise_recall = 0.0
            quant50_prediction_wise_recall = 0.0
            quant75_prediction_wise_recall = 0.0
        else:
            quant25_prediction_wise_recall = float(np.quantile(prediction_wise_rec, q=0.25))
            quant50_prediction_wise_recall = float(np.quantile(prediction_wise_rec, q=0.5))
            quant75_prediction_wise_recall = float(np.quantile(prediction_wise_rec, q=0.75))

        groundtruth_max_dices = []
        groundtruth_max_dice_precision = []
        groundtruth_max_dice_recall = []
        groundtruths_matched = []
        for groundtruth in groundtruth_results:
            dice_size_pairs = groundtruth[
                "dice_size_intersect_union_precision_recall_pairs"
            ]  # List of Tuples [dice_val, groundtruth_size]
            remaining_pairs = [
                (dice, size, n_in, n_un, prec, rec)
                for dice, size, n_in, n_un, prec, rec in dice_size_pairs
                if (size > i and dice >= true_positive_dice_threshold)
            ]
            if len(remaining_pairs) == 0:
                groundtruths_matched.append(0)
            else:
                groundtruths_matched.append(1)

            remaining_non_zero_matches = [
                (dice, size, n_in, n_un, prec, rec)
                for dice, size, n_in, n_un, prec, rec in dice_size_pairs
                if (size > i and dice != 0.0)
            ]
            remaining_dices = [a[0] for a in remaining_non_zero_matches]

            if len(remaining_non_zero_matches) == 0:
                groundtruth_max_dices.append(0.0)
                groundtruth_max_dice_precision.append(0.0)
                groundtruth_max_dice_recall.append(0.0)
            else:
                remaining_max_dice_index = np.argmax(remaining_dices)
                groundtruth_max_dices.append(remaining_non_zero_matches[remaining_max_dice_index][0])
                groundtruth_max_dice_precision.append(remaining_non_zero_matches[remaining_max_dice_index][4])
                groundtruth_max_dice_recall.append(remaining_non_zero_matches[remaining_max_dice_index][5])

        ## Groundtruth Stuff
        # Dice
        mean_groundtruth_dice = float(np.mean(groundtruth_max_dices))  # Mean that shit.
        std_groundtruth_dice = float(np.std(groundtruth_max_dices))
        if len(groundtruth_max_dices) == 0.0:
            quant25_groundtruth_dice = 0.0
            quant50_groundtruth_dice = 0.0
            quant75_groundtruth_dice = 0.0
        else:
            quant25_groundtruth_dice = float(np.quantile(groundtruth_max_dices, 0.25))
            quant50_groundtruth_dice = float(np.quantile(groundtruth_max_dices, 0.5))
            quant75_groundtruth_dice = float(np.quantile(groundtruth_max_dices, 0.75))

        # Precision
        mean_groundtruth_precision = float(np.mean(groundtruth_max_dice_precision))  # Mean that shit.
        std_groundtruth_precision = float(np.std(groundtruth_max_dice_precision))
        if len(groundtruth_max_dice_precision) == 0:
            quant25_groundtruth_precision = float(np.quantile(groundtruth_max_dice_precision, 0.25))
            quant50_groundtruth_precision = float(np.quantile(groundtruth_max_dice_precision, 0.5))
            quant75_groundtruth_precision = float(np.quantile(groundtruth_max_dice_precision, 0.75))
        else:
            quant25_groundtruth_precision = 0.0
            quant50_groundtruth_precision = 0.0
            quant75_groundtruth_precision = 0.0

        # Recall
        mean_groundtruth_recall = float(np.mean(groundtruth_max_dice_recall))  # Mean that shit.
        std_groundtruth_recall = float(np.std(groundtruth_max_dice_recall))
        if len(groundtruth_max_dice_recall) == 0:
            quant25_groundtruth_recall = 0.0
            quant50_groundtruth_recall = 0.0
            quant75_groundtruth_recall = 0.0
        else:
            quant25_groundtruth_recall = float(np.quantile(groundtruth_max_dice_recall, 0.25))
            quant50_groundtruth_recall = float(np.quantile(groundtruth_max_dice_recall, 0.5))
            quant75_groundtruth_recall = float(np.quantile(groundtruth_max_dice_recall, 0.75))

        total_matched_groundtruths = sum(groundtruths_matched)
        assert len(groundtruths_matched) == n_total_groundtruths, "Something went wrong!"
        not_matched_groundtruths = len(groundtruths_matched) - total_matched_groundtruths

        if (true_positives + false_positives) == 0:
            precision = np.NaN
        else:
            precision = true_positives / (true_positives + false_positives)
        if n_total_groundtruths != 0:
            recall = true_positives / n_total_groundtruths
        else:
            recall = np.NaN
        if (precision == np.NaN) or (recall == np.NaN):
            f1_score = np.NaN
        elif (precision + recall) == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        del groundtruth_max_dices

        ## Samplewise analysis

        samplewise_mean_pd_dice = []
        samplewise_mean_gt_dice = []
        # samplewise_mean_pd_precision = []
        # samplewise_mean_pd_recall = []
        # samplewise_mean_gt_precision = []
        # samplewise_mean_gt_recall = []

        for sample in samples:
            sample_wise_remaining_predictions = [
                prediction
                for prediction in prediction_results
                if (prediction["prediction_size"] > i) and (prediction["image_name"] == sample)
            ]
            remaining_groundtruths = [
                groundtruth for groundtruth in groundtruth_results if groundtruth["image_name"] == sample
            ]

            if len(sample_wise_remaining_predictions) == 0 and len(remaining_groundtruths) == 0:
                continue
            elif len(sample_wise_remaining_predictions) == 0 and len(remaining_groundtruths) != 0:
                samplewise_mean_gt_dice.append(0.0)
                # The predictions do not exist, so they have no dice with anything (no value appended)
            elif len(sample_wise_remaining_predictions) != 0 and len(remaining_groundtruths) == 0:
                samplewise_mean_pd_dice.append(0.0)
                # The groundtruths do not exist, so they have no dice with anything (no value appended)
            else:
                prediction_wise_dice = [
                    remaining_prediction["max_dice"] for remaining_prediction in sample_wise_remaining_predictions
                ]
                samplewise_mean_pd_dice.append(float(np.mean(prediction_wise_dice)))

                groundtruth_max_dices = []
                for groundtruth in remaining_groundtruths:
                    dice_size_pairs = groundtruth["dice_size_intersect_union_precision_recall_pairs"]
                    remaining_dices = [
                        dice for dice, size, n_in, n_un, prec, rec in dice_size_pairs if (size > i and dice != 0.0)
                    ]  # Check if big enough prediction remains with non zero dice
                    if (
                        len(remaining_dices) == 0
                    ):  # No prediction remains -> 0 dice for current gt instance, else max dice
                        groundtruth_max_dices.append(0)
                    else:
                        groundtruth_max_dices.append(max(remaining_dices))
                samplewise_mean_gt_dice.append(float(np.mean(groundtruth_max_dices)))

        mean_samplewise_mean_instancewise_pd_dice = float(np.mean(samplewise_mean_pd_dice))
        std_samplewise_mean_instancewise_pd_dice = float(np.std(samplewise_mean_pd_dice))
        median_samplewise_mean_instancewise_pd_dice = float(np.median(samplewise_mean_pd_dice))
        mean_samplewise_mean_instancewise_gt_dice = float(np.mean(samplewise_mean_gt_dice))
        std_samplewise_mean_instancewise_gt_dice = float(np.std(samplewise_mean_gt_dice))
        median_samplewise_mean_instancewise_gt_dice = float(np.median(samplewise_mean_gt_dice))

        size_wise_matching_results.append(
            {
                "prediction_filter_size": i,
                "correct_predictions": true_positives,
                "false_predictions": false_positives,
                "total_predictions": n_total_predictions,
                "filtered_predictions": removed_predictions,
                "remaining_predictions": n_remaining_predictions,
                "found_groundtruths": total_matched_groundtruths,
                "not_found_groundtruths": not_matched_groundtruths,
                "total_groundtruths": n_total_groundtruths,
                "number_of_fps_per_example": float(false_positives) / float(n_samples),
                "number_of_fns_per_example": float(not_matched_groundtruths) / (float(n_samples)),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                ## Prediction stuff instance wise statistics
                "mean_datasetwise_instancewise_prediction_dice": mean_prediction_dice,
                "std_datasetwise_instancewise_prediction_dice": std_prediction_dice,
                "q25_datasetwise_instancewise_prediction_dice": quant25_prediction_dice,
                "median_datasetwise_instancewise_prediction_dice": quant50_prediction_dice,
                "q75_datasetwise_instancewise_prediction_dice": quant75_prediction_dice,
                "mean_datasetwise_instancewise_prediction_precision": mean_prediction_wise_precision,
                "std_datasetwise_instancewise_prediction_precision": std_prediction_wise_precision,
                "q25_datasetwise_instancewise_prediction_precision": quant25_prediction_wise_precision,
                "median_datasetwise_instancewise_prediction_precision": quant50_prediction_wise_precision,
                "q75_datasetwise_instancewise_prediction_precision": quant75_prediction_wise_precision,
                "mean_datasetwise_instancewise_prediction_recall": mean_prediction_wise_recall,
                "std_datasetwise_instancewise_prediction_recall": std_prediction_wise_recall,
                "q25_datasetwise_instancewise_prediction_recall": quant25_prediction_wise_recall,
                "median_datasetwise_instancewise_prediction_recall": quant50_prediction_wise_recall,
                "q75_datasetwise_instancewise_prediction_recall": quant75_prediction_wise_recall,
                ## Groundtruth stuff instance wise statistics
                "mean_datasetwise_instancewise_groundtruth_dice": mean_groundtruth_dice,
                "std_datasetwise_instancewise_groundtruth_dice": std_groundtruth_dice,
                "q25_datasetwise_instancewise_groundtruth_dice": quant25_groundtruth_dice,
                "median_datasetwise_instancewise_groundtruth_dice": quant50_groundtruth_dice,
                "q75_datasetwise_instancewise_groundtruth_dice": quant75_groundtruth_dice,
                "mean_datasetwise_instancewise_groundtruth_precision": mean_groundtruth_precision,
                "std_datasetwise_instancewise_groundtruth_precision": std_groundtruth_precision,
                "q25_datasetwise_instancewise_groundtruth_precision": quant25_groundtruth_precision,
                "median_datasetwise_instancewise_groundtruth_precision": quant50_groundtruth_precision,
                "q75_datasetwise_instancewise_groundtruth_precision": quant75_groundtruth_precision,
                "mean_datasetwise_instancewise_groundtruth_recall": mean_groundtruth_recall,
                "std_datasetwise_instancewise_groundtruth_recall": std_groundtruth_recall,
                "q25_datasetwise_instancewise_groundtruth_recall": quant25_groundtruth_recall,
                "median_datasetwise_instancewise_groundtruth_recall": quant50_groundtruth_recall,
                "q75_datasetwise_instancewise_groundtruth_recall": quant75_groundtruth_recall,
                ## Samplewise Stuff. -- For whatever this is even used atm.
                "mean_casewise_mean_instancewise_prediction_dice": mean_samplewise_mean_instancewise_pd_dice,
                "std_casewise_mean_instancewise_prediction_dice": std_samplewise_mean_instancewise_pd_dice,
                "median_casewise_mean_instancewise_prediction_dice": median_samplewise_mean_instancewise_pd_dice,
                "mean_casewise_mean_instancewise_groundtruth_dice": mean_samplewise_mean_instancewise_gt_dice,
                "std_casewise_mean_instancewise_groundtruth_dice": std_samplewise_mean_instancewise_gt_dice,
                "median_casewise_mean_instancewise_groundtruth_dice": median_samplewise_mean_instancewise_gt_dice,
            }
        )

    return size_wise_matching_results


#  deprecated Now integrated in size_wise_matching
def size_wise_dice_of_predictions(prediction_results: list[dict]):
    mean_dice_prediction_wise = []
    for i in range(200):
        remaining_predictions = [prediction for prediction in prediction_results if prediction["prediction_size"] > i]
        prediction_wise_dice = [remaining_prediction["max_dice"] for remaining_prediction in remaining_predictions]
        mean_dice = float(np.mean(prediction_wise_dice))
        mean_dice_prediction_wise.append(dict(mean_pd_dice=mean_dice, prediction_filter_size=i))

    return mean_dice_prediction_wise


#  deprecated Now integrated in size_wise_matching
def size_wise_dice_of_groundtruths(groundtruth_results: list[dict]):
    mean_dice_groundtruth_wise = []
    for i in range(200):
        groundtruth_max_dices = []
        for groundtruth in groundtruth_results:
            dice_size_pairs = groundtruth[
                "'dice_size_intersect_union_precision_recall_pairs'"
            ]  # List of Tuples [dice_val, groundtruth_size]
            remaining_dices = [dice for dice, size in dice_size_pairs if (size > i and dice != 0.0)]
            if len(remaining_dices) == 0:
                groundtruth_max_dices.append(0.0)
            else:
                groundtruth_max_dices.append(max(remaining_dices))
        mean_dice_of_size = float(np.mean(groundtruth_max_dices))  # Mean that shit.
        mean_dice_groundtruth_wise.append(dict(mean_gt_dice=mean_dice_of_size, prediction_filter_size=i))
    return mean_dice_groundtruth_wise


def average_dice_per_groundtruth_size(groundtruth_results: list[dict]):
    """Calculates the average dice of groundtruths of the same size.
    Additionally counts the instances of groundtruths of that size.

    :param groundtruth_results:
    :return:
    """
    average_dice_and_count_per_size = []
    for i in range(200):
        if i < 199:
            groundtruths_of_that_size = [
                groundtruth for groundtruth in groundtruth_results if groundtruth["groundtruth_size"] == i
            ]
        elif i == 199:
            groundtruths_of_that_size = [
                groundtruth for groundtruth in groundtruth_results if groundtruth["groundtruth_size"] >= i
            ]
        else:
            raise ValueError("Should not reach this ever")

        n_gt_of_that_size = len(groundtruths_of_that_size)
        if n_gt_of_that_size == 0:
            mean_dice = -1
        else:
            dices = []
            for remaining_groundtruth in groundtruths_of_that_size:
                dice_size_pairs = remaining_groundtruth[
                    "dice_size_intersect_union_precision_recall_pairs"
                ]  # List of Tuples [dice_val, groundtruth_size]
                remaining_dices = [dice for dice, size, _, _, _, _ in dice_size_pairs if (size > i and dice != 0.0)]
                if len(remaining_dices) == 0:
                    dices.append(0.0)
                else:
                    dices.append(max(remaining_dices))
            mean_dice = float(np.mean(dices))
        average_dice_and_count_per_size.append(dict(size=i, n_groundtruth=n_gt_of_that_size, mean_dice=mean_dice))

    return average_dice_and_count_per_size


def average_dice_per_prediction_size(prediction_results: list[dict]):
    """Calculates the average dice of the predictions of the same size.
    Additionally counts the instances of that size.


    :param prediction_results:
    :return: Dict containing the mean_dice, prediction_size, and number of predictions of the current size n_prediction
    """
    average_dice_and_count_per_size = []
    for i in range(200):
        if i < 199:
            remaining_predictions = [
                prediction for prediction in prediction_results if prediction["prediction_size"] == i
            ]
        elif i == 199:
            remaining_predictions = [
                prediction for prediction in prediction_results if prediction["prediction_size"] >= i
            ]
        else:
            raise ValueError("Should not reach this ever.")
        n_predictions_of_that_size = len(remaining_predictions)
        if n_predictions_of_that_size == 0:
            mean_dice = -1.0
        else:
            dice_of_these_predictions = [prediction["max_dice"] for prediction in remaining_predictions]
            mean_dice = float(np.mean(dice_of_these_predictions))
        size_wise_result = dict(
            mean_dice=mean_dice,
            prediction_size=i,
            n_prediction=n_predictions_of_that_size,
        )
        average_dice_and_count_per_size.append(size_wise_result)

    return average_dice_and_count_per_size
