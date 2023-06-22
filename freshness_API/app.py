from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)

# Memuat model
model = load_model('model/Model.h5')

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Mengatur ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_file):
    img = image.load_img(image_file, color_mode="rgb",
                         target_size=(150, 150), interpolation="nearest")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    classes = model.predict(img, batch_size=10)
    label = np.where(classes[0] > 0.5, 1, 0)

    if label == 0:
        return 'Fresh Fruit', 1.0 - classes[0]
    else:
        return 'Rotten Fruit', classes[0]


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(filename)

        label, confidence = predict(filename)
        result = {
            'label': label,
            'confidence': float(confidence) * 100
        }
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
