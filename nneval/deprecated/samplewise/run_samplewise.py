import os
from pathlib import Path
from nneval.deprecated.samplewise.samplewise_evaluation import calculate_samplewise_results, get_samplewise_statistics
from nneval.utils.configuration import DatasetEvalInfo
import pandas as pd
from batchgenerators.utilities import file_and_folder_operations as ff


def run_samplewise(data_pair: DatasetEvalInfo, n_processes: int = 1):
    samplewise_results = calculate_samplewise_results(data_pair, n_processes=n_processes)
    df = pd.DataFrame(samplewise_results)
    Path(data_pair.samplewise_result_p).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(data_pair.samplewise_result_p, "samplewise_dices.csv"))

    ff.save_json(
        get_samplewise_statistics(samplewise_results),
        os.path.join(data_pair.samplewise_result_p, "casewise_means_noresample.json"),
    )
