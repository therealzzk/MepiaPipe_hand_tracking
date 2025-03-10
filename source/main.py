import os
import numpy as np
import torch
from flask import Flask, request, jsonify
import tempfile
import utils
from LSTMmodel import CNNLSTMModel

app = Flask(__name__)

num_classes = 64
output_size = 128
lstm_hidden_size = 128
lstm_num_layers = 2
dropout = 0.2
learning_rate = 0.0001
batch_size = 32
num_epochs = 70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMModel(num_classes=num_classes,
                     output_size=output_size,
                     lstm_hidden_size=lstm_hidden_size,
                     lstm_num_layers=lstm_num_layers,
                     dropout=dropout)

file_path = "hand_skeleton_mode_test.pth"
if os.path.exists(file_path):
    print("File exists")
else:
    print("File does not exist")

model.load_state_dict(torch.load("hand_skeleton_mode_test.pth"))
model.eval()

def process_keypoint(poses, lefHand, rightHand):
    #conbine 3 matrix together
    num_frames = poses.shape[0]
    poses_flat = poses.reshape(num_frames, -1)  # Shape: (num_frames, 33 * 3)
    lefHand_flat = lefHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)
    rightHand_flat = rightHand.reshape(num_frames, -1)  # Shape: (num_frames, 21 * 3)

    #process video skeleton, fix the frame number
    face_point = utils.preprocess_skeleton(poses_flat)
    left_hand_point = utils.preprocess_skeleton(lefHand_flat)
    right_hand_point = utils.preprocess_skeleton(rightHand_flat)

    combined_data = np.concatenate([left_hand_point, right_hand_point], axis=1)  # Shape: (num_frames, 2*21*3)
    face_point = torch.tensor(face_point, dtype=torch.float32)
    hand_point = torch.tensor(combined_data, dtype=torch.float32)
    return face_point, hand_point

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]

    temp_video_path = os.path.join(tempfile.gettempdir(), video_file.filename)
    video_file.save(temp_video_path)
    
    keypoints = utils.getSkeleton(temp_video_path)
    all_poses, all_left_hands, all_right_hands = utils.json_to_array(keypoints)
    pose_points, hand_points = process_keypoint(all_poses, all_left_hands, all_right_hands)
    pose_points = pose_points.unsqueeze(0)
    hand_points = hand_points.unsqueeze(0)
    with torch.no_grad():
        output = model(pose_points, hand_points).cpu()
        _, predicted = torch.max(output, 1)
        word = utils.getWordById(predicted[0].item())
    return jsonify({"prediction": word})

if __name__ == "__main__":
    app.run(debug=True)