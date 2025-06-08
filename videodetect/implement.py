import os
import cv2
import face_recognition
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN

datadir= "dataset" #dataset directory
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.xlsx"
OUTPUT_VIDEO = "output_with_faces.mp4"

def encode_faces():
    known_encodings = []
    known_names = []

    for person in os.listdir(datadir):
        person_dir = os.path.join(datadir, person)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person)
                    print(f"[INFO] Encoded: {image_path}")
                else:
                    print(f"[WARNING] No face found in: {image_path}")
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")

    data = {"encodings": known_encodings, "names": known_names}
    pd.to_pickle(data, ENCODINGS_FILE)
    print(f"[SUCCESS] Encodings saved to {ENCODINGS_FILE}")

# load the encodings from the file
def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] No encodings found. Please run encode_faces() first.")
        return None
    return pd.read_pickle(ENCODINGS_FILE)

#this is attendance marking function
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

# this function recognizes faces in a video and saves the output with recognized faces highlighted
def recognize_and_save(input_video='video1.mp4'):
    detector = MTCNN()
    data = load_encodings()
    if data is None:
        return

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    # === Setup VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("[INFO] Processing video. Press 'q' to quit preview window...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = []
        for face in faces:
            x, y, w, h = face['box']
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))

        # Encode faces
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match = face_distances.argmin()
            name = "Unknown"
            if matches[best_match]:
                name = known_names[best_match]
                mark_attendance(name)

            # here drawing rectangle and name on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # here we write the frame to the output video
        out.write(frame)

        # we show here in the real-time the video with recognized faces but its not mandatory
        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video saved to {OUTPUT_VIDEO}")
# if we add new faces to the dataset, we need to run this function to encode them
#encode_faces()
#This function will recognize faces in the video and save the output with recognized faces highlighted
recognize_and_save("video1.mp4")

