import os
import uuid
import tempfile
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from torch.utils.data import DataLoader
import uvicorn

from source.LSTMmodel import CNNLSTMModel
from source.dataset import HandSkeleton
from source import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 64
output_size = 128
lstm_hidden_size = 128
lstm_num_layers = 2
dropout = 0.2
learning_rate = 0.0001
batch_size = 32
num_epochs = 70

# Load the trained model
model = CNNLSTMModel(
    num_classes=num_classes,
    output_size=output_size,
    lstm_hidden_size=lstm_hidden_size,
    lstm_num_layers=lstm_num_layers,
    dropout=dropout
).to(device)

model_path = os.path.join(os.path.dirname(__file__), "hand_skeleton_model_xiaohe_1.pth")
print("Looking for model at:", os.path.abspath(model_path))

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    raise FileNotFoundError("Model file not found!")

app = FastAPI(title="Hand Skeleton Inference API")
app.mount("/static", StaticFiles(directory="source/static"), name="static")

# Process input keypoints from pose and hand skeletons
def process_keypoint(poses, leftHand, rightHand):
    num_frames = poses.shape[0]
    poses_flat = poses.reshape(num_frames, -1)
    leftHand_flat = leftHand.reshape(num_frames, -1)
    rightHand_flat = rightHand.reshape(num_frames, -1)

    face_point = utils.preprocess_skeleton(poses_flat)
    left_hand_point = utils.preprocess_skeleton(leftHand_flat)
    right_hand_point = utils.preprocess_skeleton(rightHand_flat)

    combined_data = np.concatenate([left_hand_point, right_hand_point], axis=1)
    face_point = torch.tensor(face_point, dtype=torch.float32).to(device)
    hand_point = torch.tensor(combined_data, dtype=torch.float32).to(device)
    return face_point, hand_point

@app.get("/")
def read_index():
    return FileResponse("source/static/index.html")

# Define prediction endpoint
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    print("[API] Received video request")
    try:
        # Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
        print("[INFO] Video saved at:", tmp_path)

        # Extract skeleton keypoints from the video
        keypoints = utils.getSkeleton(tmp_path)
        print("[INFO] Keypoints extracted")

        all_poses, all_left_hands, all_right_hands = utils.json_to_array(keypoints)
        pose_points, hand_points = process_keypoint(all_poses, all_left_hands, all_right_hands)
        pose_points = pose_points.unsqueeze(0)
        hand_points = hand_points.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(pose_points, hand_points).cpu()
            _, predicted = torch.max(output, 1)
            word = utils.getWordById(predicted[0].item())

        print("[RESULT] Prediction:", word)
        return JSONResponse(content={"prediction": word})

    except Exception as e:
        print("[ERROR]", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)


