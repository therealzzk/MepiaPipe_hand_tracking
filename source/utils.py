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

def addWordToVideo(word, input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 files

    # Define the output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2
    mpHands = mp.solutions.hands
    mpHolistic = mp.solutions.holistic

    hands = mpHands.Hands()
    holistic = mpHolistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    mpDraw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames

        # Calculate text position (bottom-center)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(imgRGB)
        hand_result = hands.process(imgRGB)
        text_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height - 10  # 10 pixels above the bottom edge

        # Add the text to the frame
        cv2.putText(frame, word, (text_x, text_y), font, font_scale, color, thickness)

        #add skeleton
        if result.pose_landmarks:
            mpDraw.draw_landmarks(frame, result.pose_landmarks, mpHolistic.POSE_CONNECTIONS)

        #draw hand
        if hand_result.multi_hand_landmarks:
            for hand_id, handLms in enumerate(hand_result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame,handLms, mpHands.HAND_CONNECTIONS)
        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def getWordById(id):
    id_str = f"{(id + 1):02d}"
    json_file_path = "label.json"
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    # Search for the matching ID
    for entry in json_data:
        if entry["ID"] == id_str:
            return entry["Name"]
    
    # return f"No entry found for ID {id}"


def getSkeleton(video_name, path_to_file = "test_data"):
    video_path = os.path.join(path_to_file, f'{video_name}')
    # skeleton_data_path = f'skeleton_test_data/{video_name.replace(".mp4", ".json")}'
    skeleton_data_path = f'{video_name.replace(".mp4", ".json")}'
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
        return data


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

        max_distance_pose = np.linalg.norm(pose_array, axis=(1), keepdims=True).max(axis=1)
        max_distance_left_hand = np.linalg.norm(left_hand_array, axis=(1), keepdims=True).max(axis=1)
        max_distance_right_hand = np.linalg.norm(right_hand_array, axis=(1), keepdims=True).max(axis=1)

        pose_array = pose_array / (max_distance_pose[:, None] + 1e-8)
        left_hand_array = left_hand_array / (max_distance_left_hand[:, None] + 1e-8)
        right_hand_array = right_hand_array / (max_distance_right_hand[:, None] + 1e-8)

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