import cv2

def detect_multiple_faces(video_source=0):
    # Load the Haar cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Open video source: 0 for webcam, or file path (e.g., "video.mp4")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    while True:
        # Read one frame from the video source
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect all faces in the current frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw green rectangles around all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Multiple Face Detection - Press Q to Quit', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video source and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Use webcam (0) or a video file path (e.g., "people.mp4")
detect_multiple_faces(0)
