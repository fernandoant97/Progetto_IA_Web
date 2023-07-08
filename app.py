import time
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
import io
import shutil
import os
import uuid

app = Flask(__name__)

# Carica il modello salvato
model1 = tf.keras.models.load_model('modello_cnn_ResNet50_256.h5')
model2 = tf.keras.models.load_model('modello_cnn_MobileNetV3_256.h5')
model3 = tf.keras.models.load_model('modello_cnn_MobileNetV3_256.h5')
model4 = tf.keras.models.load_model('modello_cnn_MobileNetV3_256.h5')

# Dimensioni desiderate per l'input del modello
SIZE_X = 256
SIZE_Y = 256

def make_prediction(model, image):
    # Carica l'immagine utilizzando PIL
    img = Image.open(image)

    # Ridimensiona l'immagine alle dimensioni desiderate
    img = img.resize((SIZE_X, SIZE_Y))

    # Preprocessa l'immagine per adattarla al formato di input del modello
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)

    # Effettua la previsione utilizzando il modello
    prediction = model.predict(img)

    # Determina il risultato della previsione
    result = 'Buono' if prediction < 0.5 else 'Cattivo'

    return result

@app.route('/')
def index():
    return render_template('index.html')

def delete_old_images():
    folder = 'static/images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


@app.route('/predict', methods=['POST'])
def predict():
    # Delete any old images
    delete_old_images()
    # Ricevi l'immagine caricata dall'utente nel form
    image = request.files['image']

    # Verifica se è stata caricata un'immagine
    if not image.filename:
        error_message = "Nessuna immagine selezionata. Carica un'immagine e riprova."
        return render_template('index.html',prediction=None, error_message=error_message)

    # Salva l'immagine in una cartella temporanea
    image_path = 'static/images/' + image.filename
    image.save(image_path)
    image_url = '/static/images/' + image.filename

    # Effettua le previsioni con i 4 modelli
    predictions = []
    predictions.append(make_prediction(model1, image_path))
    predictions.append(make_prediction(model2, image_path))
    predictions.append(make_prediction(model3, image_path))
    predictions.append(make_prediction(model4, image_path))

    # Passa l'URL dell'immagine e il risultato della previsione al template
    return render_template('index.html', prediction=predictions, image_url=image_url)

@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    # Delete any old images
    delete_old_images()
    data = request.get_json()
    image_data = data['image']
    
    if "," in image_data:
        image_data = base64.b64decode(image_data.split(",")[1])
    else:
        # Gestisci il caso in cui non c'è una virgola nella stringa di dati dell'immagine
        # Potresti, ad esempio, decodificare l'intera stringa, assumendo che sia già codificata in base64
        image_data = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    # Genera un nome univoco per l'immagine utilizzando uuid
    image_name = f'webcam_image_{uuid.uuid4().hex}.png'
    
    # Salva l'immagine con il nuovo nome univoco
    image.save(f'static/images/{image_name}')
    image_url = f'/static/images/{image_name}'  # Aggiorna il percorso dell'immagine

    predictions = []
    predictions.append(make_prediction(model1, f'static/images/{image_name}'))
    predictions.append(make_prediction(model2, f'static/images/{image_name}'))
    predictions.append(make_prediction(model3, f'static/images/{image_name}'))
    predictions.append(make_prediction(model4, f'static/images/{image_name}'))

    return jsonify({'predictions': predictions, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
