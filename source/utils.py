import json
import cv2
import numpy as np


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result


def get_label_by_filename(filename):
    if filename.endswith(".json"):
        file_id = filename.split("_")[0]
        label = int(file_id)-1

    return label

MAX_FRAME_NUM = 120
def preprocess_skeleton(skeleton, max_frames = MAX_FRAME_NUM):
    num_frames = skeleton.shape[0]
    feature_size = skeleton.shape[1]
    if num_frames <= max_frames:
        padding = np.zeros((max_frames - num_frames, feature_size))  # Padding with zeros
        precessed_data = np.vstack([skeleton, padding])  # Stack the padding at the end
    else:
        interval = num_frames // max_frames
        precessed_data = skeleton[::interval][:max_frames]
    
    return precessed_data

#translate joint json file to np array
POSE_NUM_POINTS = 33  # MediaPipe Holistic has 33 pose landmarks
HAND_NUM_POINTS = 21  # MediaPipe Hands has 21 landmarks per hand

def process_json_to_arrays(json_file_path):
    with open(json_file_path, "r") as file:
            data = json.load(file)

     # Initialize arrays to store the results
    all_poses = []
    all_left_hands = []
    all_right_hands = []

    for frame_data in data:
        # Pose landmarks
        if "pose" in frame_data and frame_data["pose"]:
            pose_array = np.array(frame_data["pose"])[:, :3]  # Shape: (33, 3)
        else:
            pose_array = np.zeros((POSE_NUM_POINTS, 3))  # Shape: (33, 3)

        # Hand landmarks
        left_hand_array = np.zeros((HAND_NUM_POINTS, 3))  # Default zero array
        right_hand_array = np.zeros((HAND_NUM_POINTS, 3))  # Default zero array

        if "hands" in frame_data and frame_data["hands"]:
            if len(frame_data["hands"]) >= 1:
                left_hand_array = np.array(frame_data["hands"][0])  # First hand
            if len(frame_data["hands"]) == 2:
                right_hand_array = np.array(frame_data["hands"][1])  # Second hand
        
        pose_anchor = pose_array[0]  # First point in pose
        left_hand_anchor = left_hand_array[0]  # First point in left hand
        right_hand_anchor = right_hand_array[0]  # First point in right hand

        pose_array -= pose_anchor  # Normalize pose
        left_hand_array -= left_hand_anchor  # Normalize left hand
        right_hand_array -= right_hand_anchor

        # Append arrays to results
        all_poses.append(pose_array)
        all_left_hands.append(left_hand_array)
        all_right_hands.append(right_hand_array)
        

    # Convert lists of arrays to NumPy arrays for easier handling
    all_poses = np.array(all_poses)  # Shape: (num_frames, 33, 3)
    all_left_hands = np.array(all_left_hands)  # Shape: (num_frames, 21, 3)
    all_right_hands = np.array(all_right_hands)  # Shape: (num_frames, 21, 3)

    return all_poses, all_left_hands, all_right_hands




# json_file_path = "C:\\Users\\Xiaohe\\Downloads\\SignLanguageRecognition\\lsa64_raw\skeleton_data\\036_001_004.json"  # Update with your JSON file path
# poses, left_hands, right_hands = process_json_to_arrays(json_file_path)

# print("Pose data shape:", poses.shape)          # (num_frames, 33, 4)
# print("Left hand data shape:", left_hands.shape)  # (num_frames, 21, 3)
# print("Right hand data shape:", right_hands.shape) # (num_frames, 21, 3)

# print("Pose data :", poses)          # (num_frames, 33, 4)
# print("Left hand data :", left_hands)  # (num_frames, 21, 3)
# print("Right hand data :", right_hands) # (num_frames, 21, 3)