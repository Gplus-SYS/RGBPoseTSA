import numpy as np  

class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic. You can choose to insert keyframes in the downsampling clip.

    Required keys are "total_frames", added or modified keys
    are "samplekey", "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
        keyframes(int): the nums of keyframes. Default: 0.

    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=0,
                 keyframes = 0,
                 regular = False):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        self.keyframes = keyframes
        self.regular = regular
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

    def _get_train_clips(self, num_frames, clip_len, results):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset
            inds = inds + off

            inds = np.sort(inds)

            num_frames = old_num_frames
            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len, results):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset
            inds = inds + off
            
            inds = np.sort(inds)
            all_inds.append(inds)
            num_frames = old_num_frames

        return np.concatenate(all_inds)
    



    def __call__(self, results):
        
        if self.regular:
            if type(results) == dict:
                num_frames = results['total_frames']
                start_frame = 0
                end_frame = num_frames - 1
                inds = np.linspace(start_frame, end_frame, self.clip_len)
                inds = np.mod(inds, num_frames)
                inds = inds
                results['frame_inds'] = inds.astype(int)
                results['clip_len'] = self.clip_len
                results['frame_interval'] = None
                results['num_clips'] = self.num_clips
                return results
            elif type(results) == list:
                for result in results:
                    num_frames = result['total_frames']
                    start_frame = 0
                    end_frame = num_frames - 1
                    inds = np.linspace(start_frame, end_frame, self.clip_len)
                    result['frame_inds'] = inds.astype(int)
                    result['clip_len'] = self.clip_len
                    result['frame_interval'] = None
                    result['num_clips'] = self.num_clips
                return results

        else:
            if type(results) == dict:
                num_frames = results['total_frames']

                if results.get('test_mode', False):
                    inds = self._get_test_clips(num_frames, self.clip_len, results)
                else:
                    inds = self._get_train_clips(num_frames, self.clip_len, results)
                if np.array([0]) not in inds:
                    inds =  inds = np.concatenate((np.array([0]), inds[:-1]))
                inds = np.mod(inds, num_frames)
                inds = inds
                results['frame_inds'] = inds.astype(int)
                results['clip_len'] = self.clip_len
                results['frame_interval'] = None
                results['num_clips'] = self.num_clips
                return results
            elif type(results) == list:
                for result in results:
                    num_frames = result['total_frames']
                    if result.get('test_mode', False):
                        inds = self._get_test_clips(num_frames, self.clip_len, result)
                    else:
                        inds = self._get_train_clips(num_frames, self.clip_len, result)
                    if np.array([0]) not in inds:
                        inds =  inds = np.concatenate((np.array([0]), inds[:-1]))
                    inds = np.mod(inds, num_frames)
                    inds = inds
                    result['frame_inds'] = inds.astype(int)
                    result['clip_len'] = self.clip_len
                    result['frame_interval'] = None
                    result['num_clips'] = self.num_clips
                return results


