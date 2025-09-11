import os
import time
import pickle
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import mediapipe as mp
from imutils import rotate_bound


def load_and_encode(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.uint8)
    encs = face_recognition.face_encodings(img)
    return encs


# ---------- CONFIG ----------
DATASET_DIR = "Dataset"
ENCODINGS_FILE = "encodings.pkl"
IMG_EXTS = (".jpg", ".jpeg", ".png")
RECOGNITION_INTERVAL_FRAMES = 5
TRACKER_RESET_FRAMES = 30
FACE_DISTANCE_THRESHOLD = 0.45
COOLDOWN_SECONDS = 30
SCALE_FACTOR = 0.5
# -----------------------------

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.55)


# ---------------- utility functions ----------------
def save_encodings(encodings, names, path=ENCODINGS_FILE):
    with open(path, "wb") as f:
        pickle.dump((encodings, names), f)
    print("[INFO] Encodings saved.")


def load_encodings(path=ENCODINGS_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None, None


def align_face_by_eyes(face_img_rgb):
    landmarks_list = face_recognition.face_landmarks(face_img_rgb)
    if not landmarks_list:
        return np.array(face_img_rgb, dtype=np.uint8)
    lm = landmarks_list[0]
    if 'left_eye' not in lm or 'right_eye' not in lm:
        return np.array(face_img_rgb, dtype=np.uint8)

    left_eye = np.mean(lm['left_eye'], axis=0)
    right_eye = np.mean(lm['right_eye'], axis=0)
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    rotated = rotate_bound(face_img_rgb, -angle)
    return np.array(rotated, dtype=np.uint8)


def crop_with_margin(frame, x1, y1, x2, y2, margin=0.2):
    h, w = frame.shape[:2]
    w_box = x2 - x1
    h_box = y2 - y1
    mx = int(w_box * margin)
    my = int(h_box * margin)
    xa = max(0, x1 - mx)
    ya = max(0, y1 - my)
    xb = min(w, x2 + mx)
    yb = min(h, y2 + my)
    return frame[ya:yb, xa:xb]


def mark_attendance(name, filename_prefix="attendance"):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{filename_prefix}_{today}.csv"
    ts = datetime.now().strftime("%H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f'{name},{today},{ts}\n')
    print(f"[ATTENDANCE] {name} at {today} {ts}")


# ---------------- Build / load encodings ----------------
encodings, names = load_encodings()
if encodings is None:
    print("[INFO] Building encodings from dataset...")
    encodings = []
    names = []
    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if fname.lower().endswith(IMG_EXTS):
                path = os.path.join(person_dir, fname)

                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.array(img, dtype=np.uint8)

                found = face_recognition.face_encodings(img)
                if not found:
                    continue
                encodings.append(found[0])
                names.append(person)

                for angle in (-15, 15):
                    try:
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                        rotated = cv2.warpAffine(img, M, (w, h))
                        rotated = np.array(rotated, dtype=np.uint8)
                        r_enc = face_recognition.face_encodings(rotated)
                        if r_enc:
                            encodings.append(r_enc[0])
                            names.append(person)
                    except Exception:
                        pass
    save_encodings(encodings, names)

print(f"[INFO] Total encodings: {len(encodings)} for {len(set(names))} people")

# --------------- Runtime: capture + recognition --------------
video = cv2.VideoCapture(0)
frame_count = 0
trackers = []
labels = []
last_seen = {}
print("[INFO] Starting. Press Q to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    orig = frame.copy()
    ih, iw = frame.shape[:2]

    if frame_count % TRACKER_RESET_FRAMES == 0:
        trackers = []
        labels = []

    if frame_count % RECOGNITION_INTERVAL_FRAMES == 0:
        trackers = []
        labels = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * iw)
                y1 = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x2, y2 = x1 + w, y1 + h
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)

                crop = crop_with_margin(rgb, x1, y1, x2, y2, margin=0.25)
                if crop.size == 0:
                    continue

                try:
                    aligned = align_face_by_eyes(crop)
                except Exception:
                    aligned = crop

                aligned = np.array(aligned, dtype=np.uint8)
                encs = face_recognition.face_encodings(aligned)

                name = "Unknown"
                if encs:
                    enc = encs[0]
                    dists = face_recognition.face_distance(encodings, enc)
                    if len(dists) > 0:
                        best_idx = np.argmin(dists)
                        best_dist = dists[best_idx]
                        if best_dist <= FACE_DISTANCE_THRESHOLD:
                            name = names[best_idx]
                            now_ts = time.time()
                            if name not in last_seen or (now_ts - last_seen[name]) > COOLDOWN_SECONDS:
                                mark_attendance(name)
                                last_seen[name] = now_ts

                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(orig, (x1, y1, w, h))
                trackers.append(tracker)
                labels.append(name)

    else:
        new_trackers = []
        new_labels = []
        for tr, label in zip(trackers, labels):
            ok, box = tr.update(orig)
            if not ok:
                continue
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            new_trackers.append(tr)
            new_labels.append(label)
        trackers, labels = new_trackers, new_labels

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
