import os
import pickle
from flask import Flask, request
import json
from PIL import Image
import io
import utils_2
import cv2


app = Flask(__name__)

@app.route('/', methods=["post"])
def _predict():
    data = {"Upload files": False}
    if request.files["image"] :
        image = request.files["image"]
        # name = request.form["name"]
        name = image.filename
        image_path = os.path.join(app.root_path, 'image_predict/' + str(name))
        image.save(image_path)
        image_pre = cv2.imread(image_path)
        face = utils_2._crop_face_224(image_pre)
        face_rz = cv2.resize(face, (224,224))
        face_rz_gray = cv2.cvtColor(face_rz, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(image_path, face_rz)        
        
        # image = Image.open(io.BytesIO(image)).convert("L")
        
        
        # image_face = utils_2._crop_face_224(image)
        
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recognizers/face-trainner.yml")
        
        labels = {"person_name": 1}
        with open("pickles/face-labels.pickle", "rb") as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
        
        id_, conf = recognizer.predict(face_rz_gray)
        if conf >= 20:
            predicted = labels[id_]
            data["Name"]= predicted  
            data["Confidence"] = round(conf,2)
        else: data["Name"] = "Can not recognition"
        data["Upload files"] = True
        data["Path"] = image_path
    return json.dumps(data, ensure_ascii=False, cls=utils_2.NumpyEncoder)


if __name__ == "__main__":
    print("Application is running...!")
    app.run(host="127.0.0.1")