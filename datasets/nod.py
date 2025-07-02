from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class NODDataset(CocoDataset):
    """Dataset for NOD."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60),(106, 0, 228),]
    }
