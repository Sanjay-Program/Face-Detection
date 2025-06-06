from mtcnn import MTCNN
import cv2

detect = MTCNN()
image = cv2.cvtColor(cv2.imread('example.jpeg'), cv2.COLOR_BGR2RGB)
output = detect.detect_faces(image)

for i in output:
    x, y, w, h = i['box']
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('MTCNN Face Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)