import cv2
import numpy as np

def detect_faces_optimized(image_path):
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    left_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Read and prepare image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Detect right profiles (original profile cascade)
    profile_faces = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(40, 40)
    )
    
    # Detect left profiles (flip image for left profiles)
    flipped_gray = cv2.flip(gray, 1)
    left_faces = left_profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(40, 40)
    )
    # Convert coordinates back to original image space
    left_faces = [(image.shape[1]-x-w, y, w, h) for (x, y, w, h) in left_faces]
    
    # Combine all detections
    all_faces = list(frontal_faces) + list(profile_faces) + left_faces
    
    # Apply non-maximum suppression
    all_faces = non_max_suppression(np.array(all_faces), overlapThresh=0.3)
    
    # Draw results
    for (x, y, w, h) in all_faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Optimized Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression(boxes, overlapThresh):
    # Implementation of non-maximum suppression
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]
    
    area = (boxes[:,2]) * (boxes[:,3])
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

# Usage
detect_faces_optimized('example.jpeg')