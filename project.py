import os
import time
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
from scipy.spatial.distance import cosine

# ---------- CONFIG ----------
DATASET_DIR = "Dataset"          # Dataset/person1/, Dataset/person2/
COOLDOWN_SECONDS = 30
MODEL_NAME = "SFace"           # Alternatives: Facenet512, VGG-Face, Dlib
DIST_THRESHOLD = 0.4             # tune threshold: lower = stricter
# -----------------------------

# ---------------- Attendance function ----------------
def mark_attendance(name, filename_prefix="attendance"):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{filename_prefix}_{today}.csv"
    ts = datetime.now().strftime("%H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{name},{today},{ts}\n")
    print(f"[ATTENDANCE] {name} at {today} {ts}")

# ---------------- Precompute embeddings ----------------
print("[INFO] Building database embeddings...")
db_embeddings = []
db_labels = []

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for fname in os.listdir(person_dir):
        path = os.path.join(person_dir, fname)
        try:
            reps = DeepFace.represent(img_path=path, model_name=MODEL_NAME,
                                      enforce_detection=False)
            if reps:
                db_embeddings.append(reps[0]["embedding"])
                db_labels.append(person)
        except Exception as e:
            print("[WARN] Could not process", path, e)

print(f"[INFO] Stored {len(db_embeddings)} embeddings for {len(set(db_labels))} people.")

# ---------------- Runtime loop ----------------
video = cv2.VideoCapture(0)
last_seen = {}
print("[INFO] Starting DeepFace Attendance. Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    try:
        detections = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn", enforce_detection=False)

        for det in detections:
            fa = det["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

            # crop face
            face_img = frame[y:y+h, x:x+w]

            reps = DeepFace.represent(img_path=face_img, model_name=MODEL_NAME,
                                      enforce_detection=False)

            if reps:
                embedding = reps[0]["embedding"]

                # Compare with database
                best_label = "Unknown"
                best_score = 1.0
                for db_emb, label in zip(db_embeddings, db_labels):
                    score = cosine(embedding, db_emb)
                    if score < best_score:
                        best_score = score
                        best_label = label

                # Threshold + Attendance
                if best_score < DIST_THRESHOLD:
                    now_ts = time.time()
                    if best_label not in last_seen or (now_ts - last_seen[best_label]) > COOLDOWN_SECONDS:
                        mark_attendance(best_label)
                        last_seen[best_label] = now_ts
                else:
                    best_label = "Unknown"

                # Draw rectangle + label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, best_label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        print("[ERROR] Runtime:", e)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
