from pathlib import Path
import SimpleITK as sitk
import numpy as np


def main():
    tr_path = "/mnt/cluster-data-all/t006d/nnunetv1/nnUNet_raw_data/Task136_BrainMetsShare/imagesTr"
    lbl_path = "/mnt/cluster-data-all/t006d/nnunetv1/nnUNet_raw_data/Task136_BrainMetsShare/labelsTr"
    ts_path = "/mnt/cluster-data-all/t006d/nnunetv1/nnUNet_raw_data/Task136_BrainMetsShare/imagesTs"
    for p in [tr_path, lbl_path, ts_path]:
        for tr_file in Path(p).glob("*.nii.gz"):
            im = sitk.ReadImage(str(tr_file))
            # inplane: 0.94, throughplane: 1.0
            spacing = im.GetSpacing()
            if np.allclose(spacing, np.array((0.94, 0.94, 1.0))):
                continue
            else:
                print(f"Spacing of {tr_file} is {im.GetSpacing()}")
                im.SetSpacing((0.94, 0.94, 1.0))
                sitk.WriteImage(im, str(tr_file))


if __name__ == "__main__":
    main()
