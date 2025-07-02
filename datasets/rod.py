from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class RODDataset(CocoDataset):
    """Dataset for ROD."""

    METAINFO = {
        'classes':
        ('Tram', 'Car', 'Truck', 'Cyclist', 'Pedestrian', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60), (106, 0, 228), (60, 20, 60), (106, 50, 228),]
    }

    