import json
import os
import cv2
import mediapipe as mp



# cap = cv2.VideoCapture(1) #using camera
# cap = cv2.VideoCapture(0) # using laptop camera
# video_name = '036_001_004'
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'test_data')
# video_name = ""

for video_name in os.listdir(data_dir):
    
    video_path = os.path.join(data_dir, f'{video_name}')

    output_video_path =  f'skeleton_test_video/{video_name}'# Path to save the output video
    skeleton_data_path = f'skeleton_test_data/{video_name.replace(".mp4", ".json")}'

    if os.path.isfile(video_path):
        print(f"Video path is valid: {video_path}")
    else:
        print(f"File not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    mpHands = mp.solutions.hands
    mpHolistic = mp.solutions.holistic

    hands = mpHands.Hands()
    holistic = mpHolistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    mpDraw = mp.solutions.drawing_utils

    data = []
    ## capture hand from camera
    try:
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
                # file.write(f"Frame {frame_count + 1}:\n")

                #draw pose
                if result.pose_landmarks:
                    for point_id, lm in enumerate(result.pose_landmarks.landmark):
                        x, y, z, visibility = lm.x, lm.y, lm.z, lm.visibility
                        frame_data["pose"].append([lm.x, lm.y, lm.z, lm.visibility])
                    mpDraw.draw_landmarks(frame, result.pose_landmarks, mpHolistic.POSE_CONNECTIONS)

                #draw hand
                if hand_result.multi_hand_landmarks:
                    for hand_id, handLms in enumerate(hand_result.multi_hand_landmarks):
                        hand_points = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
                        frame_data["hands"].append(hand_points)
                        mpDraw.draw_landmarks(frame,handLms, mpHands.HAND_CONNECTIONS)

                # Add frame data to the dataset
                data.append(frame_data)
                out.write(frame)
                frame_count += 1
                # cv2.imshow('Processed Video', frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break
        print(f"Processed {frame_count} frames.")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved as: {output_video_path}")

        # Save collected data as JSON
        with open(skeleton_data_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Skeleton data saved as: {skeleton_data_path}")

print("Done")