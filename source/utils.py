import json
import os
import cv2
import numpy as np
import mediapipe as mp


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def getSkeleton(video_name, path_to_file = "test_data"):
    video_path = os.path.join(path_to_file, f'{video_name}')
    skeleton_data_path = f'skeleton_test_data/{video_name.replace(".mp4", ".json")}'
    if os.path.isfile(video_path):
        print(f"Video path is valid: {video_path}")
    else:
        print(f"File not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        exit()

    mpHands = mp.solutions.hands
    mpHolistic = mp.solutions.holistic
    hands = mpHands.Hands()
    holistic = mpHolistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    data = []
    with open(skeleton_data_path, "w") as file:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error in reading frames.")
                break
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(imgRGB)
            hand_result = hands.process(imgRGB)
            frame_data = {"frame": frame_count, "pose": [], "hands": []}

            #draw pose
            if result.pose_landmarks:
                for point_id, lm in enumerate(result.pose_landmarks.landmark):
                    # x, y, z, visibility = lm.x, lm.y, lm.z, lm.visibility
                    frame_data["pose"].append([lm.x, lm.y, lm.z, lm.visibility])

            #draw hand
            if hand_result.multi_hand_landmarks:
                for hand_id, handLms in enumerate(hand_result.multi_hand_landmarks):
                    hand_points = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
                    frame_data["hands"].append(hand_points)

            data.append(frame_data)
            frame_count += 1

        cap.release()
        with open(skeleton_data_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Skeleton data saved as: {skeleton_data_path}")


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
    all_poses, all_left_hands, all_right_hands = json_to_array(data)
    return all_poses, all_left_hands, all_right_hands

def json_to_array(data):
    all_poses = []
    all_left_hands = []
    all_right_hands = []

    for frame_data in data:
        # Pose landmarks
        if "pose" in frame_data and frame_data["pose"]:
            pose_array = np.array(frame_data["pose"])  # Shape: (33, 4)
        else:
            pose_array = np.zeros((POSE_NUM_POINTS, 4))  # Shape: (33, 4)

        # Hand landmarks
        left_hand_array = np.zeros((HAND_NUM_POINTS, 3))  # Default zero array
        right_hand_array = np.zeros((HAND_NUM_POINTS, 3))  # Default zero array

        if "hands" in frame_data and frame_data["hands"]:
            if len(frame_data["hands"]) >= 1:
                left_hand_array = np.array(frame_data["hands"][0])  # First hand
            if len(frame_data["hands"]) == 2:
                right_hand_array = np.array(frame_data["hands"][1])  # Second hand

        # Append arrays to results
        all_poses.append(pose_array)
        all_left_hands.append(left_hand_array)
        all_right_hands.append(right_hand_array)
        

    # Convert lists of arrays to NumPy arrays for easier handling
    all_poses = np.array(all_poses)  # Shape: (num_frames, 33, 4)
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