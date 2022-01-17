import cv2
import pickle
# faces_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('image_predict/1.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faces_cascade.detectMultiScale(img_gray, scaleFactor = 1.5, minNeighbors=5)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
# cv2.imshow('Gray image', img)
# cv2.waitKey(0)

def _crop_face_224(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x,y,w,h) in faces:
        img_face = img[y: y + h, x: x + w]
    
    # cv2.imwrite(img_path + img_name, img_face)
    return img_face


# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_test= _crop_face_224(img)
face = cv2.resize(image_test, (224,224))
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
recognizer = cv2.face.LBPHFaceRecognition_create()
recognizer.read("recognizers/face-trainer.yml")
labels = {"person": 1}
with open("pickle/face-labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
id_, conf = recognizer.predict(image_test)
if conf >= 50 and conf <=95:
    name = labels[id_]

print(name)
cv2.imshow('crop image',image_test)
cv2.waitKey(0)
