import cv2
import numpy as np

def detect_challenging_faces(image_path):
    # Load multiple cascade classifiers
    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    ]
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    all_faces = []
    
    # Multi-stage detection with different parameters
    for cascade in cascades:
        # Standard detection
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,  # Lowered for better sensitivity
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(list(faces))
        
        # More sensitive detection for difficult cases
        faces_sensitive = cascade.detectMultiScale(
            gray,
            scaleFactor=1.02,  # Finer scaling
            minNeighbors=3,    # More sensitive
            minSize=(20, 20),  # Smaller faces
            maxSize=(300, 300)  # Prevent oversized detections
        )
        all_faces.extend(list(faces_sensitive))
    
    # Focused detection on bottom-right quadrant
    height, width = gray.shape
    roi = gray[height//2:height, width//2:width]
    
    for cascade in cascades:
        faces_roi = cascade.detectMultiScale(
            roi,
            scaleFactor=1.03,
            minNeighbors=3,
            minSize=(25, 25)
        )
        # Convert ROI coordinates to full image coordinates
        faces_roi = [(x+width//2, y+height//2, w, h) for (x, y, w, h) in faces_roi]
        all_faces.extend(faces_roi)
    
    # Non-maximum suppression with lower threshold
    all_faces = non_max_suppression(np.array(all_faces), overlapThresh=0.15)
    
    # Draw results
    for (x, y, w, h) in all_faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Enhanced Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression(boxes, overlapThresh):
    # ... (same implementation as before) ...
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
detect_challenging_faces('example.jpeg')