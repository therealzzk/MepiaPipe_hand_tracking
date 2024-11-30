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

import utils 



class HandSkeleton(Dataset):
    def __init__(self, data_dir, train = True):
        super(HandSkeleton, self).__init__()
        self.data_dir = data_dir
        self.train = train

        self.skeleton_files = [
            f for f in os.listdir(os.path.join(self.data_dir, 'skeleton_data'))
            if f.endswith('.json') and os.path.getsize(os.path.join(self.data_dir, 'skeleton_data', f)) > 0
        ]
        # self.label_files = os.path.join(self.data_dir, 'labels.json')

    def __getitem__(self, idx):
        filename = self.skeleton_files[idx]
        skeleton_path = os.path.join(self.data_dir, 'skeleton_data/', filename)
        poses, lefHand, rightHand = utils.process_json_to_arrays(skeleton_path)

        #conbine 3 matrix together
        num_frames = poses.shape[0]
        poses_flat = poses.reshape(num_frames, -1)  # Shape: (num_frames, 33 * 4)
        lefHand_flat = lefHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        rightHand_flat = rightHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        combined_data = np.concatenate([lefHand_flat, rightHand_flat], axis=1)  # Shape: (num_frames, 33*4 + 2*21*3)

        label = utils.get_label_by_filename(filename)

        #process video skeleton, fix the frame number
        face_point = utils.preprocess_skeleton(poses_flat)
        hand_point = utils.preprocess_skeleton(combined_data)

        face_point = torch.tensor(face_point, dtype=torch.float32)
        hand_point = torch.tensor(hand_point, dtype=torch.float32)
        return face_point, hand_point, label


    def __len__(self):
        return len(self.skeleton_files)



