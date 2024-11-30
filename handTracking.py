import cv2
import mediapipe as mp
import time
import os

video_path = "data/001_001_001.mp4"
cap = cv2.VideoCapture(video_path)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (0,0,255), thickness = 5)    # hand landmark
handConStyle = mpDraw.DrawingSpec(color = (0,255,0 ), thickness = 10)  # hand connection
pTime = 0
cTime = 0
frame_counter = 0

save_dir = "output_images"
os.makedirs(save_dir, exist_ok=True)

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



