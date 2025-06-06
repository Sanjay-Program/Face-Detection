import cv2

def detect_faces_dnn(video_source=0, confidence_threshold=0.5):
    # Load pre-trained DNN face detector model from OpenCV
    model_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = cv2.data.haarcascades.replace("haarcascades", "dnn") + "deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Use OpenCV's VideoCapture (webcam or file)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Convert frame to a blob for DNN processing
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        # Loop over all detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence*100:.1f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the output
        cv2.imshow("DNN Face Detection - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Use 0 for webcam or a video file like "video.mp4"
detect_faces_dnn(0)
