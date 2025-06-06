import cv2

def detect_faces_in_video(video_source=0):
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Open video source (0 = default webcam, or provide a file path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with rectangles
        cv2.imshow('Face Detection - Press Q to Quit', frame)

        # Press 'q' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

# Run face detection on webcam
detect_faces_in_video()
