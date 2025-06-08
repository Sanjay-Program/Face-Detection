import cv2
import face_recognition
import os
from datetime import datetime

# Load known faces
known_face_encodings = []
known_face_names = []

for file in os.listdir("known_faces"):
    image = face_recognition.load_image_file(f"known_faces/{file}")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(file)[0])  # filename without extension

def log_recognition(name):
    with open("recognition_log.csv", "a") as f:
        f.write(f"{name},{datetime.now()}\n")

def recognize_faces_dnn(video_source=0, confidence_threshold=0.5):
    model_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

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

                # Extract face ROI
                face_roi = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb_face)
                name = "Unknown"

                if encodings:
                    match = face_recognition.compare_faces(known_face_encodings, encodings[0])
                    face_distances = face_recognition.face_distance(known_face_encodings, encodings[0])
                    best_match_index = face_distances.argmin()

                    if match[best_match_index]:
                        name = known_face_names[best_match_index]
                        log_recognition(name)

                # Draw and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_faces_dnn(0)
