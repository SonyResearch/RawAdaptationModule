# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import imageio
import numpy as np
from mmcv.transforms import LoadImageFromFile, BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadNumpyFromFile(LoadImageFromFile):
    # inherit from LoadImageFromFile e.g. for get_loading_pipeline() detection

    """Load an image from file.
    Required Keys:
    - img_path
    Modified Keys:
    - img
    - img_shape
    - ori_shape
    """

    def __init__(self,
                 to_float32: bool = True,
                 ) -> None:
        super().__init__()
        self.to_float32 = to_float32

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.
        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if isinstance(results, str):
            results = dict(img_path=results)

        filename = results['img_path']

        if filename.lower().endswith(('.jpg', '.png', '.bmp')):
            img = imageio.v2.imread(filename)
        else:
            img = np.load(filename)

        if self.to_float32:
            img = img.astype(np.float32)


        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, ')

        return repr_str


@TRANSFORMS.register_module()
class RGGBtoRGB(BaseTransform):
    """RGGBtoRGB transformation.

    Required Keys:

    - img (np.uint8)


    Modified Keys:

    - img (np.uint8)

    """

    def __init__(self,) -> None:
        pass

    def transform(self, results: dict) -> dict:
        img = results["img"]
        if img.shape[2] == 3:
            return results

        img_r = img[..., 0]
        img_g1 = img[..., 1]
        img_g2 = img[..., 2]
        img_b = img[..., 3]

        img_g = (img_g1 + img_g2) / 2

        img = np.stack((img_r, img_g, img_b), axis=2)

        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
