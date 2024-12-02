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
from scipy.interpolate import interp1d

def time_warp(face_points, left_hand_point, right_hand_point, sigma=0.05):
    original_t = np.linspace(0, 1, face_points.shape[0])  # Sequence length
    distorted_t = original_t + np.random.normal(0, sigma, size=original_t.shape)
    distorted_t = np.clip(distorted_t, 0, 1)  # Ensure time stays in range

    distorted_t, unique_indices = np.unique(distorted_t, return_index=True)
    face_points = face_points[unique_indices]
    left_hand_point = left_hand_point[unique_indices]
    right_hand_point = right_hand_point[unique_indices]

    # Interpolate face points and hand points
    interp_face = interp1d(distorted_t, face_points, axis=0, kind='linear', fill_value="extrapolate")
    interp_left_hand = interp1d(distorted_t, left_hand_point, axis=0, kind='linear', fill_value="extrapolate")
    interp_right_hand = interp1d(distorted_t, right_hand_point, axis=0, kind='linear', fill_value="extrapolate")
    
    face_points_warped = interp_face(original_t)
    left_hand_points_warped = interp_left_hand(original_t)
    right_hand_points_warped = interp_right_hand(original_t)
    
    return face_points_warped, left_hand_points_warped, right_hand_points_warped

def scale_points(face_points, left_hand_point, right_hand_point):
    scale_range=(0.5, 3)
    scale_factor = np.random.uniform(*scale_range)  # Same scaling factor for all points
    face_points_scaled = face_points * scale_factor
    left_hand_points_scaled = left_hand_point * scale_factor
    right_hand_points_scaled = right_hand_point * scale_factor
    return face_points_scaled, left_hand_points_scaled, right_hand_points_scaled

def translate_points(face_points, left_hand_point, right_hand_point):
    translation_range=(-0.05, 0.05)
    translation = np.random.uniform(*translation_range, size=(1, 3))  # Same translation for all points
    face_points_translated = face_points + translation
    left_hand_points_translated = left_hand_point + translation
    right_hand_points_translated = right_hand_point + translation
    return face_points_translated, left_hand_points_translated, right_hand_points_translated

def rotate_points(face_points, left_hand_point, right_hand_point):
    angle_range=(-10, 10)
    angle = np.radians(np.random.uniform(*angle_range))  # Same angle for both
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    face_points_rotated = np.dot(face_points, rotation_matrix.T)
    left_hand_points_rotated = np.dot(left_hand_point, rotation_matrix.T)
    right_hand_points_rotated = np.dot(right_hand_point, rotation_matrix.T)
    return face_points_rotated, left_hand_points_rotated, right_hand_points_rotated

def add_noise(face_points, left_hand_point, right_hand_point):
    noise_level=0.01
    noise_face = np.random.normal(0, noise_level, size=face_points.shape)
    noise_left_hand = np.random.normal(0, noise_level, size=left_hand_point.shape)
    noise_right_hand = np.random.normal(0, noise_level, size=right_hand_point.shape)  # Generate noise
    face_points_noisy = face_points + noise_face
    left_hand_points_noisy = left_hand_point + noise_left_hand
    right_hand_points_noisy = right_hand_point + noise_right_hand
    return face_points_noisy, left_hand_points_noisy, right_hand_points_noisy

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

        label = utils.get_label_by_filename(filename)

        #process video skeleton, fix the frame number
        face_point = utils.preprocess_skeleton(poses_flat)
        left_hand_point = utils.preprocess_skeleton(lefHand_flat)
        right_hand_point = utils.preprocess_skeleton(rightHand_flat)

        if self.train:
            face_points_reshaped = face_point.reshape(-1, 33, 3)
            left_hand_point_reshaped = left_hand_point.reshape(-1, 21, 3)
            right_hand_point_reshaped = right_hand_point.reshape(-1, 21, 3)
            face_point, left_hand_point, right_hand_point = self.augment_skeleton(face_points_reshaped, left_hand_point_reshaped, right_hand_point_reshaped)
            face_point = face_point.reshape(-1, 33 * 3)
            left_hand_point = left_hand_point.reshape(-1, 21 * 3)
            right_hand_point = right_hand_point.reshape(-1, 21 * 3)

        combined_data = np.concatenate([left_hand_point, right_hand_point], axis=1)  # Shape: (num_frames, 2*21*3)
        face_point = torch.tensor(face_point, dtype=torch.float32)
        hand_point = torch.tensor(combined_data, dtype=torch.float32)
        return face_point, hand_point, label, filename


    def __len__(self):
        return len(self.skeleton_files)
    
    def augment_skeleton(self, face_point, left_hand_point, right_hand_point):
        # Apply augmentations consistently to both
        face_point, left_hand_point, right_hand_point = scale_points(face_point, left_hand_point, right_hand_point)
        face_point, left_hand_point, right_hand_point = translate_points(face_point, left_hand_point, right_hand_point)
        face_point, left_hand_point, right_hand_point = rotate_points(face_point, left_hand_point, right_hand_point)
        face_point, left_hand_point, right_hand_point = add_noise(face_point, left_hand_point, right_hand_point)
        face_point, left_hand_point, right_hand_point = time_warp(face_point, left_hand_point, right_hand_point)

        face_anchor = face_point[:, 0:1, :]  # First point in each frame (anchor point)
        face_point -= face_anchor  # Center around anchor
        max_distance_face = np.linalg.norm(face_point, axis=2, keepdims=True).max(axis=1, keepdims=True)
        face_point /= (max_distance_face + 1e-8)  # Scale to unit distance

    # Step 3: Normalize left hand points
        left_hand_anchor = left_hand_point[:, 0:1, :]  # First point in each frame (anchor point)
        left_hand_point -= left_hand_anchor  # Center around anchor
        max_distance_left_hand = np.linalg.norm(left_hand_point, axis=2, keepdims=True).max(axis=1, keepdims=True)
        left_hand_point /= (max_distance_left_hand + 1e-8)  # Scale to unit distance

    # Step 4: Normalize right hand points
        right_hand_anchor = right_hand_point[:, 0:1, :]  # First point in each frame (anchor point)
        right_hand_point -= right_hand_anchor  # Center around anchor
        max_distance_right_hand = np.linalg.norm(right_hand_point, axis=2, keepdims=True).max(axis=1, keepdims=True)
        right_hand_point /= (max_distance_right_hand + 1e-8)  # Scale to unit distance

        return face_point, left_hand_point, right_hand_point



