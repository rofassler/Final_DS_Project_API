from flask import Flask
from flask import request
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from flask import render_template

MODEL_FILE = "cnn.h5"
app = Flask(__name__)


@app.route('/form')
def form():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    images = []
    i = 0
    for f in request.files.getlist("file"):
        if f.filename != '':
            name = f"test/predict{i}.jpg"
            images.append(name)
            f.save("static/"+name)
            i += 1

    img_gen = ImageDataGenerator()
    test_data = img_gen.flow_from_directory('static', classes=['test'],
                                            batch_size=64,
                                            target_size=(224, 224),
                                            color_mode='grayscale', shuffle=True)

    pred = model.predict(test_data)
    return render_template('presentacion.html', pred=pred, images=images)


if __name__ == "__main__":
    model = load_model(MODEL_FILE)
    app.run()
