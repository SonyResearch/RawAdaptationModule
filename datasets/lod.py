from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class LODDataset(CocoDataset):
    """Dataset for LOD."""

    METAINFO = {
        'classes':
        ('car', 'motorcycle', 'bicycle', 'chair', 'dining table', 'bottle', 'tv', 'bus', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60), (106, 0, 228), (60, 20, 60), (106, 50, 228), (200, 10, 228),
            (106, 50, 0), (100, 200, 228),]
    }

    