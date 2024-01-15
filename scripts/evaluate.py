from argparse import ArgumentParser
from pathlib import Path


def evaluate_semantic_seg():
    parser = ArgumentParser()
    parser.add_argument("-pd", "--pd_dir", type=Path, required=True)
    parser.add_argument("-gt", "--gt_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)


def evaluate_instance_seg():
    parser = ArgumentParser()
    parser.add_argument("-pd", "--pd_dir", type=Path, required=True)
    parser.add_argument("-gt", "--gt_dir", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    

if __name__ == "__main__":
    print("Only entrypoint file.")
