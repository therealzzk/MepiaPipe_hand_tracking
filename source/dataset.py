import json
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import source.utils as utils



class HandSkeleton(Dataset):
    def __init__(self, data_dir, train = True):
        super(HandSkeleton, self).__init__()
        self.data_dir = data_dir
        self.train = train

        self.skeleton_files = os.listdir(os.path.join(self.data_dir, 'skeleton_data'))
        self.label_files = os.path.join(self.data_dir, 'labels.json')

    def __getitem__(self, idx):
        filename = self.skeleton_files[idx]
        skeleton_path = os.path.join(self.skeleton_files, filename)
        poses, lefHand, rightHand = utils.process_json_to_arrays(skeleton_path)

        #conbine 3 matrix together
        num_frames = poses.shape[0]
        poses_flat = poses.reshape(num_frames, -1)  # Shape: (num_frames, 33 * 4)
        lefHand_flat = lefHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        rightHand_flat = rightHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        combined_data = np.concatenate([poses_flat, lefHand_flat, rightHand_flat], axis=1)  # Shape: (num_frames, 33*4 + 2*21*3)

        label = utils.getLabelbyFilename(filename)

        #process video skeleton, fix the frame number

        data = utils.preprocess_skeleton(combined_data)
        return data, label


    def __len__(self):
        return len(self.skeleton_files)



