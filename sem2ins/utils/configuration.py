from __future__ import annotations
from dataclasses import dataclass, field

import os
from pathlib import Path

import scipy.ndimage as nd

pd_result_template = "soft_result_prediction.json"
gt_result_template = "soft_result_groundtruth.json"
hard_result_template = "hard_result.json"
prediction_groundtruth_matching_template = "gtKernel_{}_Ndilation_{}_pdKernel_{}_Ndilation_{}"
param_name_convention = "kernel_{}_Nclosing_{}"

kernel_labeling = [
    ("ball", nd.generate_binary_structure(3, 2)),
    ("cross", nd.generate_binary_structure(3, 1)),
    ("rectangle", nd.generate_binary_structure(3, 3)),
]


@dataclass
class PdGtPair:
    pd_p: str
    gt_p: str


#### New Pathing stuff: ####
class DataPair:
    def __init__(
        self,
        out_p: str,
        gt_class: int,
        pd_class: int,
        gt_p: str = None,
        pd_p: str = None,
        filtering_size: None | int = 0,
    ):
        self.gt_p = gt_p
        self.pd_p = pd_p
        self.gt_class: int = gt_class
        self.pd_class: int = pd_class
        self.filtering_size: int | None = filtering_size
        inter_naming = "GTid{}_PDid{}_eval".format(gt_class, pd_class)
        out_p = os.path.join(out_p, inter_naming)
        self._out_p = out_p
        self._resample_p_gt = os.path.join(out_p, "resampled", "gt")
        self._resample_p_pd = os.path.join(out_p, "resampled", "pd")
        self._instance_p_gt = os.path.join(out_p, "instanced", "gt")
        self._instance_p_pd = os.path.join(out_p, "instanced", "pd")
        self._samplewise_result_p = os.path.join(self._out_p, "samplewise_evaluation")
        self._instancewise_result_p = os.path.join(self._out_p, "instancewise_evaluation")

    # Location/pathing creation.
    # Highest folder: State the comparison between which classes
    #   (GTid_{}_PDid_{}_eva)
    #   | - resampled
    #   |
    #   | - instance_evaluation
    #   |
    #   | - samplewise_evaluation

    @property
    def samplewise_result_p(self):
        return self._samplewise_result_p

    @samplewise_result_p.setter
    def samplewise_result_p(self, new_path: str) -> None:
        self._samplewise_result_p = new_path
        return

    @samplewise_result_p.getter
    def samplewise_result_p(self):
        os.makedirs(self._samplewise_result_p, exist_ok=True)
        return self._samplewise_result_p

    @property
    def instancewise_result_p(self):
        return self._instancewise_result_p

    @instancewise_result_p.setter
    def instancewise_result_p(self, new_path: str) -> None:
        self._instancewise_result_p = new_path
        return

    @instancewise_result_p.getter
    def instancewise_result_p(self):
        os.makedirs(self._instancewise_result_p, exist_ok=True)
        return self._instancewise_result_p

    @property
    def result_p(self):
        return self._result_p

    @result_p.setter
    def result_p(self, new_path: str) -> None:
        self._result_p = new_path
        return

    @result_p.getter
    def result_p(self):
        os.makedirs(self._result_p, exist_ok=True)
        return self._result_p

    @property
    def out_p(self):
        return self._out_p

    @out_p.getter
    def out_p(self) -> str:
        os.makedirs(self._out_p, exist_ok=True)
        return self._out_p

    @out_p.setter
    def out_p(self, new_path: str) -> None:
        self._out_p = new_path
        return

    @property
    def resample_p_gt(self):
        return self._resample_p_gt

    @resample_p_gt.getter
    def resample_p_gt(self):
        os.makedirs(self._resample_p_gt, exist_ok=True)
        return self._resample_p_gt

    @resample_p_gt.setter
    def resample_p_gt(self, new_path: str):
        self._resample_p_gt = new_path

    @property
    def resample_p_pd(self):
        return self._resample_p_pd

    @resample_p_pd.getter
    def resample_p_pd(self):
        os.makedirs(self._resample_p_pd, exist_ok=True)
        return self._resample_p_pd

    @resample_p_pd.setter
    def resample_p_pd(self, new_path: str):
        self._resample_p_pd = new_path

    @property
    def instance_p_gt(self):
        return self._instance_p_gt

    @instance_p_gt.getter
    def instance_p_gt(self):
        os.makedirs(self._instance_p_gt, exist_ok=True)
        return self._instance_p_gt

    @instance_p_gt.setter
    def instance_p_gt(self, new_path: str):
        self._instance_p_gt = new_path

    @property
    def instance_p_pd(self):
        return self._instance_p_pd

    @instance_p_pd.getter
    def instance_p_pd(self):
        os.makedirs(self._instance_p_pd, exist_ok=True)
        return self._instance_p_pd

    @instance_p_pd.setter
    def instance_p_pd(self, new_path: str):
        self._instance_p_pd = new_path
