import os
import shutil
import uuid

from flask import Flask
from flask import request
from flask import render_template
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

MODEL_FILE = "cnn.h5"
IMG_FOLDER = "static"
app = Flask(__name__)


@app.route('/form')
def form():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    new_folder = str(uuid.uuid4())
    try:
        os.makedirs(IMG_FOLDER+"/"+new_folder)
    except ValueError:
        render_template('error.html', message="Failed to create folder, try again")

    try:
        images = []
        i = 0

        for f in request.files.getlist("file"):
            if f.filename != '':
                name = f"{new_folder}/predict{i}.jpg"
                images.append(name)
                f.save(f"{IMG_FOLDER}/{name}")
                i += 1
    except ValueError:
        render_template('error.html', message="Error saving images")

    img_gen = ImageDataGenerator()
    test_data = img_gen.flow_from_directory('static', classes=[new_folder], batch_size=64,
                                            target_size=(224, 224), color_mode='grayscale', shuffle=True)

    predictions = [round(value[0]*100, 2) for value in model.predict(test_data)]

    return render_template('show_results.html', pred=predictions, images=images)


if __name__ == "__main__":
    model = load_model(MODEL_FILE)
    app.run()
