from ultralytics import YOLO
import cv2
import requests

MAX_DISTANCE = 50
REQUIRED_FRAMES = 20

CAMERA_INDEX = 0

BASE_URL = "http://localhost:8000/api"
USERS_URL = f"{BASE_URL}/users/recognition-map/"
ATTENDANCE_URL = f"{BASE_URL}/attendance/"
ATTENDANCE_SESSION_ID = 1


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")

face_model = YOLO("models/yolov11m-face.pt").to("cuda")


response = requests.get(USERS_URL, timeout=2)
response.raise_for_status()

users = {user["id"]: user["first_name"] for user in response.json()["users"]}

print("Loaded users:")
print(users)


def send_attendance(user_id):
    try:
        response = requests.post(
            ATTENDANCE_URL,
            json={
                "user_id": user_id,
                "session_id": ATTENDANCE_SESSION_ID,
            },
            timeout=2,
        )

        if response.status_code in (200, 201):
            print(f"Attendance recorded for user {user_id}")
            return True

        print(
            f"Attendance request failed: " f"{response.status_code} - {response.text}"
        )

    except requests.RequestException as error:
        print(f"Attendance request error: {error}")

    return False


face_states = {}


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    result = face_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )[0]

    processed_image = result.plot()

    if result.boxes is None or len(result.boxes) == 0:
        cv2.imshow("capture", processed_image)

        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

        continue

    boxes = result.boxes.xyxy.cpu().int().tolist()

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().int().tolist()
    else:
        track_ids = [None] * len(boxes)

    for box, track_id in zip(boxes, track_ids):

        left, top, right, bottom = box

        height, width = frame.shape[:2]

        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        if right <= left or bottom <= top:
            continue

        face = frame[top:bottom, left:right]

        if face.size == 0:
            continue

        face = cv2.resize(face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        user_id, distance = face_recognizer.predict(face)

        user_id = int(user_id)
        distance = float(distance)

        recognized = distance <= MAX_DISTANCE and user_id in users

        if recognized:
            name = users[user_id]
        else:
            name = "Unknown"

        if track_id is not None:

            if track_id not in face_states:
                face_states[track_id] = {
                    "user_id": None,
                    "streak": 0,
                    "attendance_sent": False,
                }

            state = face_states[track_id]

            if recognized:

                if state["user_id"] == user_id:
                    state["streak"] += 1
                else:
                    state["user_id"] = user_id
                    state["streak"] = 1
                    state["attendance_sent"] = False

                if state["streak"] >= REQUIRED_FRAMES and not state["attendance_sent"]:
                    print(
                        f"CONFIRMED: "
                        f"{name} | "
                        f"ID: {user_id} | "
                        f"Distance: {distance:.2f}"
                    )

                    attendance_success = send_attendance(user_id)

                    if attendance_success:
                        state["attendance_sent"] = True

            else:
                state["user_id"] = None
                state["streak"] = 0
                state["attendance_sent"] = False

        if track_id is not None and track_id in face_states:
            streak = face_states[track_id]["streak"]
        else:
            streak = 0

        label = f"{name} | " f"{distance:.0f} | " f"{streak}/{REQUIRED_FRAMES}"

        cv2.putText(
            processed_image,
            label,
            (left, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("capture", processed_image)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()
