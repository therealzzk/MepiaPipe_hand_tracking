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

def scale_points(face_points, hand_points):
    scale_range=(0.9, 1.1)
    scale_factor = np.random.uniform(*scale_range)  # Same scaling factor for all points
    face_points_scaled = face_points * scale_factor
    hand_points_scaled = hand_points * scale_factor
    return face_points_scaled, hand_points_scaled

def translate_points(face_points, hand_points):
    translation_range=(-0.05, 0.05)
    translation = np.random.uniform(*translation_range, size=(1, 3))  # Same translation for all points
    face_points_translated = face_points + translation
    hand_points_translated = hand_points + translation
    return face_points_translated, hand_points_translated

def rotate_points(face_points, hand_points):
    angle_range=(-10, 10)
    angle = np.radians(np.random.uniform(*angle_range))  # Same angle for both
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    face_points_rotated = np.dot(face_points, rotation_matrix.T)
    hand_points_rotated = np.dot(hand_points, rotation_matrix.T)
    return face_points_rotated, hand_points_rotated

def add_noise(face_points, hand_points):
    noise_level=0.01
    noise_face = np.random.normal(0, noise_level, size=face_points.shape)
    noise_hand = np.random.normal(0, noise_level, size=hand_points.shape)  # Generate noise
    face_points_noisy = face_points + noise_face
    hand_points_noisy = hand_points + noise_hand
    return face_points_noisy, hand_points_noisy

class HandSkeleton(Dataset):
    def __init__(self, data_dir, train = True):
        super(HandSkeleton, self).__init__()
        self.data_dir = data_dir
        self.train = train

        if self.train:
            self.skeleton_files = [
                f for f in os.listdir(os.path.join(self.data_dir, 'skeleton_data'))
                if f.endswith('.json') and os.path.getsize(os.path.join(self.data_dir, 'skeleton_data', f)) > 0
            ]
        else: 
            self.skeleton_files = [
                f for f in os.listdir(os.path.join(self.data_dir, 'skeleton_test_data'))
                if f.endswith('.json') and os.path.getsize(os.path.join(self.data_dir, 'skeleton_test_data', f)) > 0
            ]
        # self.label_files = os.path.join(self.data_dir, 'labels.json')

    def __getitem__(self, idx):
        filename = self.skeleton_files[idx]
        if self.train ==True:
            skeleton_path = os.path.join(self.data_dir, 'skeleton_data/', filename)
        else: 
            skeleton_path = os.path.join(self.data_dir, 'skeleton_test_data/', filename)
        poses, lefHand, rightHand = utils.process_json_to_arrays(skeleton_path)

        #conbine 3 matrix together
        num_frames = poses.shape[0]
        poses_flat = poses.reshape(num_frames, -1)  # Shape: (num_frames, 33 * 3)
        lefHand_flat = lefHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        rightHand_flat = rightHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
        combined_data = np.concatenate([lefHand_flat, rightHand_flat], axis=1)  # Shape: (num_frames, 2*21*3)

        label = utils.get_label_by_filename(filename)

        #process video skeleton, fix the frame number
        face_point = utils.preprocess_skeleton(poses_flat)
        hand_point = utils.preprocess_skeleton(combined_data)

        if self.train:
            face_points_reshaped = face_point.reshape(-1, 33, 3)
            hand_points_reshaped = hand_point.reshape(-1, 42, 3)
            face_point, hand_point = self.augment_skeleton(face_points_reshaped, hand_points_reshaped)
            face_point = face_point.reshape(-1, 33 * 3)
            hand_point = hand_point.reshape(-1, 42 * 3)

        face_point = torch.tensor(face_point, dtype=torch.float32)
        hand_point = torch.tensor(hand_point, dtype=torch.float32)
        return face_point, hand_point, label


    def __len__(self):
        return len(self.skeleton_files)
    
    def augment_skeleton(self, face_point, hand_point):
        # Apply augmentations consistently to both
        face_point, hand_point = scale_points(face_point, hand_point)
        face_point, hand_point = translate_points(face_point, hand_point)
        face_point, hand_point = rotate_points(face_point, hand_point)
        face_point, hand_point = add_noise(face_point, hand_point)
        return face_point, hand_point



