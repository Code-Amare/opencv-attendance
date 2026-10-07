from ultralytics import YOLO
import cv2
import os
import numpy as np

face_model = YOLO("models/yolov11m-face.pt").to("cuda")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

folder_path = "train-pics/"

folders = {1: "Amare=992c2a60-f26e-4b9c-921c-993a37861b9d", 2: "Temesgen=7d6cfd41-191d-40ac-8396-7a11c25c6343", 3: "Alazar=37c452c0-4e8c-4c02-a156-239526f99e5e", 4: "Misgana=688152a4-6425-4464-8a50-e0476395e86c"}

processed_image = []
faces = []
labels = []


for label, folder in folders.items():
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