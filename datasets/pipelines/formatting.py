import torch
import numpy as np

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        data = {}
        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')


class FormatShape(object):
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format):
        self.input_format = input_format
        if self.input_format not in ['NCTHW', 'NCHW', 'NCTHW_Heatmap']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            if 'imgs' in results:
                imgs = results['imgs']
                imgs = np.transpose(imgs, (0, 3, 1, 2))
                # M x C x H x W
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape
        elif self.input_format == 'NCTHW_Heatmap':
            if 'imgs' in results:
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                imgs = results['imgs']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

        return results

class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, stream='keypoint',num_person=1, mode='zero'):
        self.stream = stream
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 
        
        keypoint = results[self.stream]
        if self.stream == 'keypoint' and'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] >= self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0,print(T,nc)
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['imgs'] = np.ascontiguousarray(keypoint)
        return results