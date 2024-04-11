from pathlib import Path
from toinstance.predict import create_instance

from pathlib import Path
from toinstance.predict import create_instance


def instance_generation(
    semantic_pd_path: str | Path,
    semantic_gt_path: str | Path,
    output_pd_path: str | Path | None = None,
    output_gt_path: str | Path | None = None,
):
    """Exemplary call to instances."""

    if output_pd_path is None:
        output_pd_path = semantic_pd_path
        assert output_pd_path.exists(), f"Output prediction path does not exist: {output_pd_path}"
        output_pd_path = Path(output_pd_path) / "nneval" / "pd_instances"
        output_pd_path.mkdir(parents=True, exist_ok=True)

    if output_gt_path is None:
        output_gt_path = semantic_gt_path
        assert output_gt_path.exists(), f"Path to groundtruth does not exist: {output_gt_path}"
        output_gt_path = Path(output_gt_path) / "nneval" / "gt_instances"
        output_gt_path.mkdir(parents=True, exist_ok=True)

    create_instance(
        input_path=Path(semantic_pd_path),
        output_dir=Path(output_pd_path),
        overwrite=True,  # Set to true to show it actually creates stuff
    )
    create_instance(
        input_path=Path(semantic_gt_path),
        output_dir=Path(output_gt_path),
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
