import cv2
import numpy as np

def high_accuracy_face_detection(image_path):
    # Load multiple cascade classifiers
    cascades = [
        ('frontal_default', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')),
        ('frontal_alt2', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')),
        ('profile', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'))
    ]
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Advanced preprocessing pipeline
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    all_detections = []
    detection_weights = []  # Track confidence per detection
    
    # Multi-model detection with weighted confidence
    for name, cascade in cascades:
        # Standard detection
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_detections.extend(list(faces))
        detection_weights.extend([1.0]*len(faces))  # Standard confidence
        
        # Focused detection on challenging areas
        if name == 'profile':
            # Extra pass for right-side profiles
            faces_right = cascade.detectMultiScale(
                gray,
                scaleFactor=1.03,
                minNeighbors=4,
                minSize=(30, 30),
                maxSize=(200, 200)
            )
            all_detections.extend(list(faces_right))
            detection_weights.extend([0.8]*len(faces_right))  # Lower confidence
            
            # Left-side profiles (flipped image)
            flipped = cv2.flip(gray, 1)
            faces_left = cascade.detectMultiScale(
                flipped,
                scaleFactor=1.03,
                minNeighbors=4,
                minSize=(30, 30)
            )
            faces_left = [(gray.shape[1]-x-w, y, w, h) for (x,y,w,h) in faces_left]
            all_detections.extend(faces_left)
            detection_weights.extend([0.8]*len(faces_left))
    
    # Focused detection on bottom-right quadrant
    height, width = gray.shape
    roi_bottom_right = gray[height//2:, width//2:]
    
    for name, cascade in cascades:
        faces_roi = cascade.detectMultiScale(
            roi_bottom_right,
            scaleFactor=1.02,
            minNeighbors=3,
            minSize=(25, 25)
        )
        faces_roi = [(x+width//2, y+height//2, w, h) for (x,y,w,h) in faces_roi]
        all_detections.extend(faces_roi)
        detection_weights.extend([1.2 if name == 'frontal_alt2' else 1.0]*len(faces_roi))
    
    # Weighted non-maximum suppression
    final_faces = weighted_nms(np.array(all_detections), np.array(detection_weights), overlapThresh=0.25)
    
    # Post-processing validation
    validated_faces = []
    for (x,y,w,h) in final_faces:
        face_roi = gray[y:y+h, x:x+w]
        # Simple validation - check for eye regions
        if w > 30 and h > 30:  # Only validate larger faces
            eye_region = face_roi[:h//2, :]
            if cv2.mean(eye_region)[0] < 200:  # Not too bright
                validated_faces.append((x,y,w,h))
        else:
            validated_faces.append((x,y,w,h))
    
    # Draw final results
    for (x,y,w,h) in validated_faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, f'Face', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow('High Accuracy Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def weighted_nms(boxes, weights, overlapThresh):
    if len(boxes) == 0:
        return []
    
    boxes = boxes.astype("float")
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]
    
    area = boxes[:,2] * boxes[:,3]
    idxs = np.argsort(weights * area)  # Sort by weighted area
    
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
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

# Usage
high_accuracy_face_detection('example.jpeg')