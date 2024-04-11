from pathlib import Path
from toinstance.predict import create_instance

from pathlib import Path
from toinstance.predict import create_instance


def instance_generation(semantic_pd_path: Path, semantic_gt_path: Path, output_pd_path: Path, output_gt_path: Path):
    """Exemplary call to instances."""
    create_instance(
        input_path=semantic_pd_path,
        output_dir=output_pd_path,
        overwrite=True,  # Set to true to show it actually creates stuff
    )
    create_instance(
        input_path=semantic_gt_path,
        output_dir=output_gt_path,
        overwrite=True,  # Set to true to show it actually creates stuff
    )


def instance_generation(semantic_pd_path: Path, semantic_gt_path: Path, output_pd_path: Path, output_gt_path: Path):
    """Exemplary call to instances."""
    create_instance(
        input_path=semantic_pd_path,
        output_dir=output_pd_path,
        overwrite=True,  # Set to true to show it actually creates stuff
    )
    create_instance(
        input_path=semantic_gt_path,
        output_dir=output_gt_path,
        overwrite=True,  # Set to true to show it actually creates stuff
    )


if __name__ == "__main__":
    cur_path = Path(__file__).parent.parent / "tests/"

    single_cls_gts = cur_path / "sem_test_data/single_class/gts"
    single_cls_pds = cur_path / "sem_test_data/single_class/preds"
    multi_cls_gts = cur_path / "sem_test_data/multi_class/gts"
    multi_cls_pds = cur_path / "sem_test_data/multi_class/preds"

    single_cls_gts_out = cur_path / "ins_test_data/single_class/gts"
    single_cls_pds_out = cur_path / "ins_test_data/single_class/preds"
    multi_cls_gts_out = cur_path / "ins_test_data/multi_class/gts"
    multi_cls_pds_out = cur_path / "ins_test_data/multi_class/preds"

    instance_generation(
        semantic_pd_path=single_cls_pds,
        semantic_gt_path=single_cls_gts,
        output_pd_path=single_cls_pds_out,
        output_gt_path=single_cls_gts_out,
    )

    instance_generation(
        semantic_pd_path=multi_cls_pds,
        semantic_gt_path=multi_cls_gts,
        output_pd_path=multi_cls_pds_out,
        output_gt_path=multi_cls_gts_out,
    )
