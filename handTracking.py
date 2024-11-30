import cv2
import mediapipe as mp
import time
import os

#video_path = "data/001_001_001.mp4"
#ap = cv2.VideoCapture(video_path)

# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color = (0,0,255), thickness = 5)    # hand landmark
# handConStyle = mpDraw.DrawingSpec(color = (0,255,0 ), thickness = 10)  # hand connection
# pTime = 0
# cTime = 0
# frame_counter = 0

data_dir = "data"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

frame_interval = 10

for video_file in os.listdir(data_dir):
    video_path = os.path.join(data_dir, video_file)
    if not video_file.lower().endswith(('.mp4')):
        print(f"Skipping non-video file: {video_file}")
        continue
    video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
    os.makedirs(video_output_dir,exist_ok=True)
                                       
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        continue
    print(f"Processing video: {video_path}")
    
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame =  cap.read()
        if not ret:
            print(f"Finished processing video: {video_file}")
            break
        if frame_count % frame_interval == 0:
            image_name = f"frame_{frame_count}.jpg"
            image_path = os.path.join(video_output_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_frame_count} images for video: {video_file}")

print("All videos processed.")

"""
while True:
    ret, img =  cap.read()
    if ret:
        frame_counter += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        imgHeight = img.shape[0]
        imgWidth  = img.shape[1]
        #print(result.multi_hand_landmarks)
        # if(result.multi_hand_landmarks):
        #         for handLms in result.multi_hand_landmarks:
        #             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
        #             for i, lm in enumerate(handLms.landmark):
        #                  xPos = int(lm.x * imgWidth)
        #                  yPos = int(lm.y * imgHeight)
        #                  #cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0, 255),2)
        #                  if(i == 4):
        #                       cv2.circle(img, (xPos, yPos), 20, (166,56,56) , cv2.FILLED)
        #                  print(i, xPos, yPos)
                     
        if frame_counter % 10 == 0:
             save_path = os.path.join(save_dir, f"frame_{frame_counter}.jpg")
             cv2.imwrite(save_path, img)

             
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:  {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0), 3)


        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

"""

