import cv2
import face_recognition
import os
from datetime import datetime
import pandas as pd

# Path to folder with known face images
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.xlsx"

# Load known faces
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"Warning: No face found in {filename}")

# Attendance marking function
def mark_attendance(name):
    date_today = datetime.now().strftime("%Y-%m-%d")

    # Create file if not exists
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time", "Date"])
        df.to_excel(ATTENDANCE_FILE, index=False)

    df = pd.read_excel(ATTENDANCE_FILE)

    # Check if already marked today
    if not ((df["Name"] == name) & (df["Date"] == date_today)).any():
        now = datetime.now().strftime("%H:%M:%S")
        new_entry = {"Name": name, "Time": now, "Date": date_today}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"Marked attendance for {name} at {now}")

# Face recognition and attendance system
def recognize_faces_dnn(video_source=0, confidence_threshold=0.5):
    # Load DNN face detector
    model_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Open webcam or video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    print("Starting camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")

                face_roi = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)

                name = "Unknown"
                if encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, encodings[0])
                    face_distances = face_recognition.face_distance(known_face_encodings, encodings[0])
                    best_match_index = face_distances.argmin()

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        mark_attendance(name)

                # Draw bounding box and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run it
recognize_faces_dnn(0)
