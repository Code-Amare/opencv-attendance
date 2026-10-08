from ultralytics import YOLO
import cv2
import os
import numpy as np

face_model = YOLO("models/yolov11m-face.pt").to("cuda")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

folder_path = "train-pics/"

folders = os.listdir("./train-pics")

user_ids = {
    int(folder.split("=")[1]): folder.split("=")[0]
    for folder in folders
}

processed_image = []
faces = []
labels = []


for label, folder in user_ids.items():
    files = os.listdir(folder_path + folder)
    for file in files:
        image = cv2.imread(folder_path + folder + "/" + file)
        face_result = face_model(image, verbose=False)
        if len(face_result[0].boxes) == 0:
            continue

        processed_image.append(face_result)
        left, top, right, bottom = face_result[0].boxes.xyxy[0].int()
        face = image[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (200, 200))
        labels.append(label)
        faces.append(face)

face_recognizer.train(faces, np.array(labels))
face_recognizer.write("face_recognizer.yml")

cv2.imshow("Result", faces[0])
cv2.waitKey(0)
cv2.destroyAllWindows()