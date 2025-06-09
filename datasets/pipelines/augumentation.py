import cv2
import random
import numpy as np
from torch.nn.modules.utils import _pair

def _combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])

class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required keys in results are "img_shape", "keypoint", add or modified keys
    are "img_shape", "keypoint", "crop_quadruple".

    Args:
        padding (float): The padding size. Default: 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Default: 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Default: None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Default: True.

    Returns:
        type: Description of returned object.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = (hw_ratio, hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0
    
    
    
    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results
        
        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple

        return results
    


class MMCompact:


    def __init__(self, 
                 padding=0.25, 
                 threshold=10, 
                 hw_ratio=1, 
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 

        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        min_x, min_y, max_x, max_y = self._get_box(kp, img_shape)

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        results['imgs'] = self._compact_images(results['imgs'], img_shape, (min_x, min_y, max_x, max_y))
        return results

    
    def _get_box(self, keypoint, img_shape):
        # will return x1, y1, x2, y2
        h, w = img_shape

        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
        return (min_x, min_y, max_x, max_y)

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        min_x, min_y, max_x, max_y = self._get_box(kp, img_shape)

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        results['imgs'] = self._compact_images(results['imgs'], img_shape, (min_x, min_y, max_x, max_y))
        return results


class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "flip_direction".
    The Flip augmentation should be placed after any cropping / reshaping
    augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
    """
    _directions = ['horizontal', 'vertical']


    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        
        assert direction in self._directions, (f'Direction {direction} is not supported. 'f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs):
        assert self.direction in ['horizontal', 'vertical', 'diagonal']
        if self.direction == 'horizontal':
            imgs = [cv2.flip(img, 1) for img in imgs]
        elif self.direction == 'vertical':
            imgs = [cv2.flip(img, 0) for img in imgs]
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

        


    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        if 'keypoint' in results:
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'])
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        return results

class Resize:
    """Resize keypoint to a specific size.

    Required keys are "img_shape", "keypoint".

    Args:
        shape (Tuple[int]): (target_width, target_height)
    """

    def __init__(self, shape):
        assert isinstance(shape, tuple), 'The range of Resize must be a tuple'
        self.shape = shape

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for img in imgs
        ]
    
    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        
        assert 'img_shape' in results
        origin_width, origin_height = results['img_shape']
        target_width, target_height = self.shape
        self.scale_factor = np.array([target_width / origin_width, target_height / origin_height],
                                     dtype=np.float32)


        if 'imgs' in results:
            results['imgs'] = self._resize_imgs(results['imgs'], target_width, target_height)
        if 'keypoint' in results:
            results['keypoint'] = self._resize_kps(results['keypoint'][...,:2], self.scale_factor)
        
        results['scale_factor'] = results['scale_factor'] * self.scale_factor
        results['img_shape'] = (target_width, target_height)
        
        return results
    

class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs".

    Args:
        size (int): The output size of the images.
    """

    def __init__(self, size):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    # def _all_box_crop(self, results, crop_bbox):
    #     """Crop the gt_bboxes and proposals in results according to crop_bbox.

    #     Args:
    #         results (dict): All information about the sample, which contain
    #             'gt_bboxes' and 'proposals' (optional).
    #         crop_bbox(np.ndarray): The bbox used to crop the original image.
    #     """
    #     results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
    #     if 'proposals' in results and results['proposals'] is not None:
    #         assert results['proposals'].shape[1] == 4
    #         results['proposals'] = self._box_crop(results['proposals'],
    #                                               crop_bbox)
    #     return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        # # Process entity boxes
        # if 'gt_bboxes' in results:
        #     results = self._all_box_crop(results, results['crop_bbox'])

        return results
    

class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        assert type(area_range) == tuple, (f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1, (area_range[0],area_range[1])
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(int)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(int)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox((img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'keypoint' in results:
            results['keypoint'] = self._crop_kps(results['keypoint'], crop_bbox)
        if 'imgs' in results:
            results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)

        # if 'gt_bboxes' in results:
        #     results = self._all_box_crop(results, results['crop_bbox'])

        return results
    

class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", added or modified
    keys are "imgs" and "img_norm_cfg".

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
    """

    def __init__(self, mean, std, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.mean_ = np.float64(self.mean.reshape(1, -1))
        self.std_ = 1 / np.float64(self.std.reshape(1, -1))
        self.to_bgr = to_bgr

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 

        n = len(results['imgs'])
        h, w, c = results['imgs'][0].shape
        imgs = np.empty((n, h, w, c), dtype=np.float32)
        for i, img in enumerate(results['imgs']):
            imgs[i] = img

        for img in imgs:
            if self.to_bgr:
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            cv2.subtract(img, self.mean_, img)  # inplace
            cv2.multiply(img, self.std_, img)  # inplace

        results['imgs'] = imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_bgr=self.to_bgr)
        return results


        