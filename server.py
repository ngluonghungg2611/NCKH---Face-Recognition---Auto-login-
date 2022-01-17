from flask import Flask, request, render_template
from PIL import Image
import cv2
import io
import utils
import json
import os
import faces_train
app = Flask(__name__)

@app.route('/')
def _hello():
    return render_template('./index.html')

@app.route("/save_image", methods=["post"])
def _save_image():
    data = {"Upload files": False}
    if request.files['image1'] or request.files['image2'] or request.files['image3'] or request.files['image4'] or request.files['image5']:
        name = request.form["nameFile"]
        os.makedirs('images/' + name)
        for i in range(1,6):
            image = request.files['image' + str(i)]
            image_path = os.path.join(app.root_path, 'images/' + str(name) + "/" + str(i) + ".jpg")
            image.save(image_path)
        
        data["Upload files"] = True
    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)


if __name__ == "__main__":
    print("Run app!")
    app.run(host="127.0.0.1")
