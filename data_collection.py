'''
For data collection, 
classID : 0 fake, 1 real
Uses cvzone FaceDetector to capture only those instances where face is detected with conf > 0.8
max_frame_count : maximum number of frames captured
captured image aspect ratio : 640, 480
'''


import os
from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

classID = 0
outputFolderPath = r'Dataset/DataCollect/Fake'
confidence = 0.8
save = True
blurThreshold = 35
camWidth, camHeight = 640, 480
floatingPoint = 6
max_frame_count = 100

os.makedirs(outputFolderPath, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()
frame_capture_count = 0

while True:
    if frame_capture_count >= max_frame_count:
        print("✅ Data collection complete.")
        break

    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read from webcam.")
        continue

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:
                offsetW = (10 / 100) * w
                offsetH = (20 / 100) * h
                x = max(int(x - offsetW), 0)
                y = max(int(y - offsetH * 3), 0)
                w = int(w + offsetW * 2)
                h = int(h + offsetH * 3.5)

                imgFace = img[y:y + h, x:x + w]
                if imgFace.size == 0:
                    continue

                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                listBlur.append(blurValue > blurThreshold)

                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10), scale=2, thickness=3)

    if save and all(listBlur) and listBlur != []:
        timeNow = str(time()).replace('.', '')
        img_path = os.path.join(outputFolderPath, f"{timeNow}.jpg")
        txt_path = os.path.join(outputFolderPath, f"{timeNow}.txt")
        cv2.imwrite(img_path, img)

        with open(txt_path, 'a') as f:
            f.writelines(listInfo)

        print(f"[INFO] Saved {img_path}")
        frame_capture_count += 1

    cv2.imshow("Image", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Manual quit.")
        break

cap.release()
cv2.destroyAllWindows()