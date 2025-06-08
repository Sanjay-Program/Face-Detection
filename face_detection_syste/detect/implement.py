# Full Face Recognition Attendance System

import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime

# === Config ===
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.xlsx"

# === Step 1: Data Collection ===
def collect_face_data(person_name, num_samples=20):
    cap = cv2.VideoCapture(0)
    save_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture Face - Press 's' to save", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
            count += 1
    cap.release()
    cv2.destroyAllWindows()

# === Step 2: Create Face Encodings ===
def encode_faces():
    known_encodings = []
    known_names = []
    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
    data = {"encodings": known_encodings, "names": known_names}
    pd.to_pickle(data, ENCODINGS_FILE)

# === Step 3: Mark Attendance ===
def mark_attendance(name):
    date_today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time", "Date"])
        df.to_excel(ATTENDANCE_FILE, index=False)
    df = pd.read_excel(ATTENDANCE_FILE)
    if not ((df["Name"] == name) & (df["Date"] == date_today)).any():
        now = datetime.now().strftime("%H:%M:%S")
        new_entry = {"Name": name, "Time": now, "Date": date_today}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"[INFO] Attendance marked for {name} at {now}")

# === Step 4: Recognition and Attendance ===
def recognize_and_mark(video_source=0):
    data = pd.read_pickle(ENCODINGS_FILE)
    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(video_source)
    print("[INFO] Starting camera... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        for encoding, (top, right, bottom, left) in zip(encodings, boxes):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match = face_distances.argmin()
            name = "Unknown"
            if matches[best_match]:
                name = known_names[best_match]
                mark_attendance(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Example Usage ===
#collect_face_data("Dhanush", 20)  # Run once per person to collect data
# encode_faces()                # Run once after data collection
recognize_and_mark(0)         # Run to start attendance system
