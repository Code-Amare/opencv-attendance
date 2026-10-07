from ultralytics import YOLO
import cv2

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")
face_model = YOLO("yolov11m-face.pt").to("cuda")

folders = {1: "Amare", 2: "Temesgen", 3: "Alazar", 4: "Misgana"}

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