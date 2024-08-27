from pathlib import Path
from toinstance.predict import create_instance

from pathlib import Path
from toinstance.predict import create_instance


def instance_generation(input_path: Path, output_path: Path, overwrite: bool = False, is_prediction: bool = False):
    if output_path is None:
        output_path = input_path
        assert output_path.exists(), f"Output prediction path does not exist: {output_path}"
        filename = "pd_instances" if is_prediction else "gt_instances"
        output_path = Path(output_path) / "nneval" / filename
        output_path.mkdir(parents=True, exist_ok=True)
    create_instance(
        input_path=Path(input_path),
        output_dir=Path(output_path),
        overwrite=overwrite,  # Set to true to show it actually creates stuff
    )


def pd_gt_instance_generation(
    semantic_pd_path: str | Path,
    semantic_gt_path: str | Path,
    output_pd_path: str | Path | None = None,
    output_gt_path: str | Path | None = None,
    overwrite: bool = True,
):
    """Exemplary call to instances."""

    instance_generation(semantic_pd_path, output_pd_path, overwrite, is_prediction=True)
    instance_generation(semantic_gt_path, output_gt_path, overwrite, is_prediction=False)


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

    pd_gt_instance_generation(
        semantic_pd_path=single_cls_pds,
        semantic_gt_path=single_cls_gts,
        output_pd_path=single_cls_pds_out,
        output_gt_path=single_cls_gts_out,
    )

    pd_gt_instance_generation(
        semantic_pd_path=multi_cls_pds,
        semantic_gt_path=multi_cls_gts,
        output_pd_path=multi_cls_pds_out,
        output_gt_path=multi_cls_gts_out,
    )
