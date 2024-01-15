from dataclasses import field, dataclass, asdict
from pathlib import Path
from typing import Sequence
import numpy as np


@dataclass(frozen=True)  # Allows hashing
class SemanticPair:
    pd_p: Path  # Full path to prediction
    gt_p: Path  # Full path to groundtruth


@dataclass
class InstancePair:
    semantic_pd_p: Path  # Full path to prediction
    instance_pd_p: Path
    semantic_gt_p: Path  # Full path to groundtruth
    instance_gt_p: Path  # Full path to groundtruth


@dataclass
class SemanticResult:
    dice: float
    iou: float
    precision: float
    recall: float
    gt_voxels: int
    pd_voxels: int
    union_voxels: int
    intersection_voxels: int
    class_id: int
    case_id: str = field(init=False)
    volume_per_voxel: float = field(init=False)
    pd_volume: float = field(init=False)
    gt_volume: float = field(init=False)
    intersection_volume: float = field(init=False)
    union_volume: float = field(init=False)
    spacing: Sequence[float] = field(init=False)
    dimensions: Sequence[int] = field(init=False)

    def add_volume_per_voxel(self, volume_per_voxel: float):
        self.volume_per_voxel = volume_per_voxel
        self.pd_volume = self.pd_voxels * volume_per_voxel
        self.gt_volume = self.gt_voxels * volume_per_voxel
        self.intersection_volume = self.intersection_voxels * volume_per_voxel
        self.union_volume = self.union_voxels * volume_per_voxel

    def get_non_meta_values(self) -> dict:
        """Returns all values that are not meta values like dimensions and spacing, allowing to calculate statistics over them."""
        non_meta_values = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, Sequence):
                non_meta_values[k] = v

    def todict(self) -> dict:
        return asdict(self)


@dataclass
class InstanceResult:
    dice: float
    iou: float
    precision: float
    recall: float
    pd_voxels: int
    gt_voxels: int
    union_voxels: int
    intersection_voxels: int
    pd_instance_index: int
    gt_instance_index: int
    match: bool
    semantic_class_id: int
    case_id: str
    volume_per_voxel: float
    spacing: Sequence[float]
    dimensions: Sequence[int]
    dice_threshold: float
    pd_volume: float = field(init=False)
    gt_volume: float = field(init=False)
    intersection_volume: float = field(init=False)
    union_volume: float = field(init=False)
    true_positive: int = field(init=False)
    false_negative: int = field(init=False)

    def __post_init__(self):
        self.pd_volume = self.pd_voxels * self.volume_per_voxel
        self.gt_volume = self.gt_voxels * self.volume_per_voxel
        self.intersection_volume = self.intersection_voxels * self.volume_per_voxel
        self.union_volume = self.union_voxels * self.volume_per_voxel
        self._determine_tp(self.dice_threshold)

    def get_non_meta_values(self) -> dict:
        """Returns all values that are not meta values like dimensions and spacing, allowing to calculate statistics over them."""
        non_meta_values = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, Sequence):
                non_meta_values[k] = v

    def _determine_tp(self, dice_threshold: float):
        if np.isnan(self.dice):  # Neither prediction nor groundtruth
            self.true_positive = np.NaN
            self.false_negative = np.NaN
        else:
            if np.isnan(self.pd_volume):  # No prediction
                self.true_positive = np.NaN  # PD does not exist so neither TP nor FP
                self.false_negative = True  # Groundtruth exists so it is false negative
            elif np.isnan(self.gt_volume):  # No groundtruth
                self.false_negative = np.NaN
                self.true_positive = False
            else:
                if self.dice >= dice_threshold:
                    self.true_positive = 1
                    self.false_negative = 0
                else:
                    self.true_positive = 0
                    self.false_negative = 1
        self.dice_threshold = dice_threshold

    def todict(self) -> dict:
        return asdict(self)


@dataclass
class Instance:
    index: int
    voxels: int
