import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Configuration
EMBEDDINGS_PATH = "face_embeddings.npz"
CAMERA_INDEX = 0
SIMILARITY_THRESHOLD = 0.50

# Load the saved face embeddings
data = np.load(EMBEDDINGS_PATH)

known_embeddings = data["embeddings"].astype(np.float32)
known_labels = data["labels"].astype(np.int32)

names = {}

for item in data["names"]:
    user_id, name = str(item).split("=", 1)
    names[int(user_id)] = name

# Normalize embeddings to ensure cosine similarity works correctly
known_embeddings /= np.maximum(
    np.linalg.norm(known_embeddings, axis=1, keepdims=True), 1e-12
)

# Initialize InsightFace
app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

# Open the camera
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("ArcFace recognition started. Press Q to quit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read camera frame.")
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Get the normalized embedding for this face
            embedding = face.normed_embedding.astype(np.float32)

            # Compute cosine similarity against every stored embedding
            similarities = known_embeddings @ embedding

            # Find the most similar registered face
            best_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_index])

            if best_similarity >= SIMILARITY_THRESHOLD:
                user_id = int(known_labels[best_index])
                name = names.get(user_id, "Unknown")
                label = f"{name} (ID: {user_id})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            # Draw the face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display identity and similarity
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Similarity: {best_similarity:.3f}",
                (x1, max(y1 - 8, 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        cv2.imshow("ArcFace Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    data.close()
