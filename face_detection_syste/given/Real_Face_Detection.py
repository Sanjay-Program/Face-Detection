from mtcnn import MTCNN  # Importing the MTCNN class from the mtcnn library
import cv2
import tensorflow as tf
detector = MTCNN()  # Creating an MTCNN detector object for face detection
cap = cv2.VideoCapture('video4.mp4')  # Loading video file; use 0 instead of filename(video3.mp4) to use webcam cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame,'ret' is bool (success), 'frame' is the image (numpy array)

    if not ret or frame is None:
        print("Failed to grab frame")  # If frame not read correctly, exit the loop
        break

    faces = detector.detect_faces(frame)  # Detect faces in the current frame
    for face in faces:
        x, y, w, h = face['box']  # Get bounding box coordinates for each detected face
        # Draw rectangle around the face: top-left (x, y), bottom-right (x+w, y+h), green color, thickness 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Real-time Face Detection', frame)  # Display the frame with detected faces

    # Wait for 1 milliSecond and check if 'q' key is pressed; if yes, exit the loop
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
