import torch
import copy
import pickle
import random
import glob
import os
import cv2



class RGBPose_Dataset(torch.utils.data.Dataset):
    def __init__(self, seed, action_number_choosing, voter_number, data_args, pipeline):
        random.seed(seed)
        self.repeat_times = data_args.repeat
        self.split = data_args.split
        self.pipeline = pipeline
        self.action_number_choosing = action_number_choosing
        self.voter_number = voter_number
        # file path
        self.data_root = data_args.data_root
        self.data_anno = self.read_pickle(data_args.ann_file)
        
        with open(data_args.train_split_pkl, 'rb') as f:
                self.train_dataset_list = pickle.load(f)
       
        self.action_number_dict = {}
        if self.split == 'train':
            self.dataset = self.train_dataset_list
        else:
            with open(data_args.test_split_pkl, 'rb') as f:
                self.test_dataset_list = pickle.load(f)
            self.dataset = self.test_dataset_list
            self.action_number_dict_test = {}

        self.choose_list = self.train_dataset_list.copy()
        if self.action_number_choosing:
            self.preprocess()
            self.check_exemplar_dict()

    def preprocess(self):
        for item in self.train_dataset_list:
            dive_number = self.data_anno.get(item).get('action_type')
            if self.action_number_dict.get(dive_number) is None:
                self.action_number_dict[dive_number] = []
            self.action_number_dict[dive_number].append(item)
        # if self.split == 'test':
        #     for item in self.test_dataset_list:
        #         dive_number = self.data_anno.get(item).get('action_type')
        #         if self.action_number_dict_test.get(dive_number) is None:
        #             self.action_number_dict_test[dive_number] = []
        #         self.action_number_dict_test[dive_number].append(item)

    def check_exemplar_dict(self):
        if self.split == 'train':
            for key in sorted(list(self.action_number_dict.keys())):
                file_list = self.action_number_dict[key]
                for item in file_list:
                    assert self.data_anno[item].get('action_type') == key
        # if self.split == 'test':
        #     for key in sorted(list(self.action_number_dict_test.keys())):
        #         file_list = self.action_number_dict_test[key]
        #         for item in file_list:
        #             assert self.data_anno[item].get('action_type') == key
    
    def load_video(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))
        video = [cv2.imread(image_list[i]) for i in range(len(image_list))]
        for img in video:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return video
    
    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data
    

    def __len__(self):
        """Get the size of the dataset."""
        return self.repeat_times * len(self.dataset)
    
    def __getitem__(self, index):
        sample_video_name = self.dataset[index % len(self.dataset)]
        video_1_ori = self.load_video(sample_video_name)
        sample = copy.deepcopy(self.data_anno.get(sample_video_name))
        assert sample['video_name'] == sample_video_name, 'the names of vedio are not match!'
        sample['imgs'] = video_1_ori
        sample['completeness'] = (sample['dive_score'] / sample['difficulty'])
        sample['test_mode'] = (self.split == 'test')
        if self.action_number_choosing:
            # choose a exemplar
            if self.split == 'train':
                file_list = copy.deepcopy(self.action_number_dict[sample.get('action_type')])
                if len(file_list) > 1:
                    file_list.pop(file_list.index(sample_video_name))
                # randomly choosing one out
                idx = random.randint(0, len(file_list) - 1)
                target_video_name = file_list[idx]
                video_2_ori = self.load_video(target_video_name)
                target = copy.deepcopy(self.data_anno.get(target_video_name))
                assert target['video_name'] == target_video_name, 'the names of vedio are not match!'
                target['imgs'] = video_2_ori
                target['completeness'] = (target['dive_score'] / target['difficulty'])
                target['test_mode'] = (self.split == 'test')
                return self.pipeline(sample), self.pipeline(target)
            else:
                train_file_list = copy.deepcopy(self.action_number_dict[sample.get('action_type')])
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]
                target_list = []
                for item in choosen_sample_list:
                    video_2_ori = self.load_video(item)
                    target = copy.deepcopy(self.data_anno.get(item))
                    assert target['video_name'] == item, 'the names of vedio are not match!'
                    target['imgs'] = video_2_ori
                    target['completeness'] = (target['dive_score'] / target['difficulty'])
                    target['test_mode'] = True
                    target_list.append(target)
                return self.pipeline(sample), [self.pipeline(target) for target in target_list]
        else:
            return self.pipeline(sample)