import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "train-pics"
OUTPUT_PATH = "face_embeddings.npz"

app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
labels = []
names = {}

for folder in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(person_path) or "=" not in folder:
        continue

    name, user_id = folder.rsplit("=", 1)
    user_id = int(user_id)

    names[user_id] = name

    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        faces = app.get(image)

        if not faces:
            print(f"No face detected: {image_path}")
            continue

        # Use the largest detected face.
        face = max(
            faces, key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        )

        embeddings.append(face.normed_embedding)
        labels.append(user_id)

        print(f"Processed: {name} - {filename}")

if not embeddings:
    raise ValueError("No face embeddings were generated.")

np.savez_compressed(
    OUTPUT_PATH,
    embeddings=np.asarray(embeddings, dtype=np.float32),
    labels=np.asarray(labels, dtype=np.int32),
    names=np.array([f"{user_id}={name}" for user_id, name in names.items()]),
)

print(f"\nTraining data saved to {OUTPUT_PATH}")
print(f"Total embeddings: {len(embeddings)}")
print(f"Registered people: {len(names)}")
