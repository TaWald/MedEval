import os
from argparse import ArgumentParser



def collect_singler_dataset_result(dataset_id: int):
    re_pattern = r"Dataset{:03d}".format(dataset_id)
    os.environ.get("nnUNet_results")


def collect_nnunet_results():
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", help="NNunet folder containing results to aggregate", required=True)
    parser.add_argument("-f", "--nnunet_folder", help="NNunet folder containing results to aggregate", required=True)
    parser.add_argument("-o", "--output_folder", help="Output folder for aggregated results", required=True)


    try:
        values = os.environ.get("nnUNet_results")
    except KeyError:
        print("Please set the environment variable nnUNet_results when aggregating nnunet dataset results.")
        exit(1)

if __name__ == "__main__":
    main()