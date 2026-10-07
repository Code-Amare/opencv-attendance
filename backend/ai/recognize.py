from ultralytics import YOLO
import cv2
import os

folders = os.listdir("./train-pics")
print(int("37c452c0-4e8c-4c02-a156-239526f99e5e"))
exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")
face_model = YOLO("models/yolov11m-face.pt").to("cuda")

users = {
    "37c452c0-4e8c-4c02-a156-239526f99e5e": "Alazar",
    "992c2a60-f26e-4b9c-921c-993a37861b9d": "Amare",
    "688152a4-6425-4464-8a50-e0476395e86c": "Misgana",
    "7d6cfd41-191d-40ac-8396-7a11c25c6343": "Temesgen",
}

max_distance = 70

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord("x"):
    _, frame = cap.read()
    face_result = face_model(frame, verbose=False)
    processed_image = face_result[0].plot()

    if len(face_result[0].boxes) == 0:
        cv2.imshow("capture", processed_image)
        continue

    boxes = face_result[0].boxes.xyxy
    for box in boxes:

        left, top, right, bottom = box.int()
        face = frame[top:bottom, left:right]
        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        label, distance = face_recognizer.predict(face)

        if distance > max_distance:
            name = "Unknown"
        else:
            name = folders[label]

        cv2.putText(
            processed_image,
            name + "|" + str(int(distance)),
            (int(left), int(bottom) + 20),
            0,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("capture", processed_image)


cv2.waitKey(5000)
cap.release()
cv2.destroyAllWindows()