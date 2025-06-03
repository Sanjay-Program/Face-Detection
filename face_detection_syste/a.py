import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications import VGGFace
from tensorflow.keras.preprocessing import image as kimage

class AdvancedFaceDetector:
    def __init__(self):
        self.detector = MTCNN(
            min_face_size=20,
            steps_threshold=[0.6, 0.7, 0.8],  # More strict thresholds
            scale_factor=0.85  # Finer scaling
        )
        self.face_validator = VGGFace(
            model='resnet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
    def detect_faces(self, image_path):
        # Read and convert image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        original = image.copy()
        
        # Enhanced MTCNN detection
        results = self.detector.detect_faces(image)
        
        # Face validation pipeline
        valid_faces = []
        for result in results:
            x, y, w, h = result['box']
            confidence = result['confidence']
            keypoints = result['keypoints']
            
            # Basic size and confidence filtering
            if w < 30 or h < 30 or confidence < 0.9:
                continue
                
            # Extract face ROI
            face_roi = image[max(0,y):y+h, max(0,x):x+w]
            
            # Deep validation
            if self.validate_face(face_roi):
                # Keypoint verification
                if self.verify_keypoints(keypoints):
                    valid_faces.append((x, y, w, h))
        
        # Non-maximum suppression
        valid_faces = self.nms(np.array(valid_faces))
        
        # Draw results
        for (x, y, w, h) in valid_faces:
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(original, f'Face {confidence:.2f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        
        return original
    
    def validate_face(self, face_roi):
        """Use VGGFace embeddings to validate face regions"""
        try:
            # Preprocess for VGGFace
            face = cv2.resize(face_roi, (224, 224))
            face = kimage.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            
            # Get embedding
            embedding = self.face_validator.predict(face)
            
            # Simple validation (can be replaced with classifier)
            return np.linalg.norm(embedding) > 0.2  # Threshold for face-like features
        except:
            return False
    
    def verify_keypoints(self, keypoints):
        """Verify facial keypoints structure"""
        required_points = ['left_eye', 'right_eye', 'nose']
        return all(k in keypoints and all(v > 0 for v in keypoints[k]) for k in required_points)
    
    def nms(self, boxes, overlap_thresh=0.3):
        """Non-maximum suppression implementation"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = x1 + boxes[:,2]
        y2 = y1 + boxes[:,3]
        
        area = boxes[:,2] * boxes[:,3]
        idxs = np.argsort(area)[::-1]
        
        pick = []
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
                np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick]

# Usage
detector = AdvancedFaceDetector()
result_image = detector.detect_faces('example.jpeg')

# Display
cv2.imshow('Ultimate Face Detection', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()