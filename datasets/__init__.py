from .kitti import KITTIDataset
from .carla import CarlaDataset
from .eisats import EISATSDataset


__datasets__ = {
    "kitti": KITTIDataset,
    "carla": CarlaDataset,
    "eisats": EISATSDataset,
}
