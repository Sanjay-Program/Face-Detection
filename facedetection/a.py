import cv2

# Load pre-trained Haar Cascade (frontal + profile for side faces)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast in low-light
    
    # Detect frontal and profile faces
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Finer scaling for small faces
        minNeighbors=6,    # Reduce false positives
        minSize=(40, 40),  # Factory workers are usually closer to the camera
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(40, 40)
    )
    return list(frontal_faces) + list(profile_faces)

# Example usage
image = cv2.imread('example.jpeg')
faces = detect_faces(image)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Factory Worker Face Detection', image)
cv2.waitKey(0)