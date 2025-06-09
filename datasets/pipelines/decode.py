import numpy as np


class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        if 'bone' in results:
            results['bone'] = self._load_kp(results['bone'], frame_inds)

        if 'cosine' in results:
            results['cosine'] = self._load_kp(results['cosine'], frame_inds)

        if 'frames_labels' in results:
            frames_labels = list(results['frames_labels'][frame_inds])
            frames_catogeries = list(set(frames_labels))
            frames_catogeries.sort(key=frames_labels.index)
            transitions = [frames_labels.index(c) for c in frames_catogeries]
            results['transits'] = np.array([transitions[1],transitions[-1]]).astype(np.float32)

        return results
    

class RGBDecode:
    """Load and decode rgb frames with given indices.

    Required keys are "imgs", "frame_inds" (optional)
    """

    @staticmethod
    def _load_imgs(imgs, frame_inds):
        return [imgs[i] for i in frame_inds]



    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        assert 'imgs' in results, 'key(imgs) is not in result!'
        results['imgs'] = self._load_imgs(results['imgs'], frame_inds)

        if 'frames_labels' in results:
            frames_labels = list(results['frames_labels'][frame_inds])
            frames_catogeries = list(set(frames_labels))
            frames_catogeries.sort(key=frames_labels.index)
            transitions = [frames_labels.index(c) for c in frames_catogeries]
            results['transits'] = np.array([transitions[1],transitions[-1]]).astype(np.float32)

        return results
    
class MMDecode:

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_imgs(imgs, frame_inds):
        return [imgs[i] for i in frame_inds]
    

    def __call__(self, results):
        if type(results) == list:
            results = [self.__call__(result) for result in results]
            return results 

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        assert 'imgs' in results, 'key(imgs) is not in result!'
        results['imgs'] = self._load_imgs(results['imgs'], frame_inds)


        assert 'keypoint' in results, 'key(keypoint) is not in result!'
        results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)


        assert 'keypoint_score' in results, 'key(keypoint_score) is not in result!'
        results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'frames_labels' in results:
            frames_labels = list(results['frames_labels'][frame_inds])
            frames_catogeries = list(set(frames_labels))
            frames_catogeries.sort(key=frames_labels.index)
            transitions = [frames_labels.index(c) for c in frames_catogeries]
            results['transits'] = np.array([transitions[1],transitions[-1]]).astype(np.float32)

        return results
    


