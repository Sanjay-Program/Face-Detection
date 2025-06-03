import cv2
import numpy as np

def detect_all_faces(image_path):
    # Load all available cascades
    cascades = [
        ('frontal', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')),
        ('profile', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')),
        ('frontal_alt', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')),
        ('frontal_alt2', cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'))
    ]
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Advanced preprocessing
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    all_faces = []
    
    # Try multiple detection approaches
    for name, cascade in cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(list(faces))
    
    # Rotated detection for profile faces
    for angle in [-30, -15, 15, 30]:
        rotated = rotate_image(gray, angle)
        faces = cascades[1][1].detectMultiScale(
            rotated,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )
        # Convert coordinates back to original
        faces = [rotate_rect_back(x,y,w,h, angle, gray.shape) for (x,y,w,h) in faces]
        all_faces.extend(faces)
    
    # Apply NMS
    all_faces = non_max_suppression(np.array(all_faces), overlapThresh=0.2)
    
    # Draw results
    for (x,y,w,h) in all_faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    
    cv2.imshow('Enhanced Detection', image)
    cv2.waitKey(0)

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def rotate_rect_back(x,y,w,h, angle, orig_shape):
    # Convert rotated rect back to original coordinates
    center = np.array(orig_shape[1::-1]) / 2
    rot_mat = cv2.getRotationMatrix2D(tuple(center), -angle, 1.0)
    pts = np.array([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])
    pts = np.hstack([pts, np.ones((4,1))]).dot(rot_mat.T)
    x1, y1 = pts.min(0).astype(int)
    x2, y2 = pts.max(0).astype(int)
    return (x1, y1, x2-x1, y2-y1)

def non_max_suppression(boxes, overlapThresh):
    # ... (same as previous implementation) ...
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
detect_all_faces('example.jpeg')