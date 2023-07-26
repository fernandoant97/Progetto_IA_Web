from datetime import datetime
from ultralytics import YOLO
import time
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import base64
import numpy as np
import io
import shutil
import os
import uuid
import torch
from yolov5 import detect
from pathlib import Path
import pandas as pd
from collections import defaultdict

app = Flask(__name__)


# Carica il modello salvato
model1 = tf.keras.models.load_model('modello_cnn_ResNet50_256.h5')
model2 = tf.keras.models.load_model('modello_cnn_MobileNetV3_256.h5')
model3 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt') 
model4 = YOLO(model=Path("yolov8x.pt")) 

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

def make_prediction_yolov5(model, image_path):
    # Carica l'immagine
    im = Image.open(image_path)

    # Fai inferenza
    results = model(im)  # Inference

    # Salviamo i risultati in un DataFrame
    df = results.pandas().xyxy[0]

    # Creiamo un nome di file univoco basato sulla data e sull'ora attuali
    file_name = 'results_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'

    # Stampa i risultati in un file di testo
    with open(file_name, 'w') as f:
        f.write(df.to_string())
      # Verifica se il DataFrame è vuoto (nessun oggetto rilevato)
    if df.empty:
        # Eliminiamo il file dopo averlo letto
        os.remove(file_name)
        return "Nessuna prugna trovata"
    
    # Funzione per ottenere la classe con la confidenza massima da un file
    def get_class_with_highest_confidence(filename):
        class_counts = defaultdict(int)
        class_confidence_sums = defaultdict(float)

        with open(filename, 'r') as file:
            lines = file.readlines()

            # Consideriamo solo le righe che contengono dati
            data_lines = [line for line in lines if not line.startswith(' ')]

            for line in data_lines:
                # I valori sono separati da spazi, quindi splittiamo la linea
                values = line.split()
                
                # Otteniamo la colonna 'confidence'
                confidence = float(values[-3])

                # Otteniamo la colonna 'name'
                name = values[-1]

                class_counts[name] += 1
                class_confidence_sums[name] += confidence

        num_classes = len(class_counts)

        if num_classes == 1:
            return list(class_counts.keys())[0]
        elif num_classes == 2:
            return max(class_confidence_sums, key=class_confidence_sums.get)
        else:
            max_count = max(class_counts.values())
            max_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(max_classes) == 1:
                return max_classes[0]
            else:
                max_class = max(max_classes, key=lambda cls: class_confidence_sums[cls])
                return max_class
    
    # Ottieni la classe con la confidenza massima
    predicted_class = get_class_with_highest_confidence(file_name)
    
    # Eliminiamo il file dopo averlo letto
    os.remove(file_name)
    
    if predicted_class == "":
        return "Nessuna prugna rilevata"

    result_str = 'Buona' if predicted_class == 'good_prune' else 'Cattiva'

    return result_str

def make_prediction_yolov8(model, image_path):
    # Carica l'immagine
    im = Image.open(image_path)

    # Fai inferenza
    model(im, save_txt=True, save_conf=True)  # Inference

    # Funzione per ottenere la classe con la confidenza massima da un file
    def get_class_with_highest_confidence(filename):
        if os.path.getsize(filename) == 0:  # Controlla se il file è vuoto
            return None
        class_counts = defaultdict(int)
        class_confidence_sums = defaultdict(float)

        with open(filename, 'r') as file:
            lines = file.readlines()

            for line in lines:
                # I valori sono separati da spazi, quindi splittiamo la linea
                values = line.split()

                # Otteniamo la colonna 'confidence'
                confidence = float(values[-1])

                # Otteniamo la colonna 'name'
                name = int(values[0])

                class_counts[name] += 1
                class_confidence_sums[name] += confidence

        num_classes = len(class_counts)

        if num_classes == 1:
            return list(class_counts.keys())[0]
        elif num_classes == 2:
            return max(class_confidence_sums, key=class_confidence_sums.get)
        else:
            max_count = max(class_counts.values())
            max_classes = [cls for cls, count in class_counts.items() if count == max_count]

            if len(max_classes) == 1:
                return max_classes[0]
            else:
                max_class = max(max_classes, key=lambda cls: class_confidence_sums[cls])
                return max_class
    
    # Ottieni il percorso del file di testo corrispondente
    base_name = os.path.basename(image_path)
    txt_file_name = os.path.splitext(base_name)[0] + '.txt'
    txt_file_path = os.path.join('runs/detect/predict/labels', txt_file_name)
    
    # Ottieni la classe con la confidenza massima
    predicted_class = get_class_with_highest_confidence(txt_file_path)
    
    # Eliminiamo il file dopo averlo letto
    shutil.rmtree('runs')

    if predicted_class is None:
        return "Nessuna prugna rilevata"

    result_str = 'Buona' if predicted_class == 1 else 'Cattiva'

    return result_str

# Funzione personalizzata per il filtro zip nel template
def custom_zip(a, b):
    return zip(a, b)

# Aggiungi la funzione custom_zip all'ambiente di Jinja2
app.jinja_env.globals['zip'] = custom_zip

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
    # Ricevi le immagini caricate dall'utente nel form
    images = request.files.getlist('image')

    # Verifica se sono state caricate delle immagini
    if len(images) == 0 or not any(image.filename for image in images):
        error_message = "Nessuna immagine selezionata. Carica delle immagini e riprova."
        return render_template('index.html', predictions=None, error_message=error_message)

    predictions = []
    image_urls = []

    for image in images:
        # Salva l'immagine in una cartella temporanea
        image.save('static/images/' + image.filename)
        image_url = '/static/images/' + image.filename
        image_urls.append(image_url)

        # Effettua le previsioni con i 4 modelli
        image_predictions = []
        
        image_predictions.append(make_prediction(model1, 'static/images/' + image.filename))
        image_predictions.append(make_prediction(model2, 'static/images/' + image.filename))
        image_predictions.append(make_prediction_yolov5(model3, 'static/images/' + image.filename))
        image_predictions.append(make_prediction_yolov8(model4, 'static/images/' + image.filename))
  

        predictions.append(image_predictions)

    # Passa gli URL delle immagini e i risultati delle previsioni al template
    return render_template('index.html', predictions=predictions, image_urls=image_urls)

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
    predictions.append(make_prediction_yolov5(model3, f'static/images/{image_name}'))
    predictions.append(make_prediction_yolov8(model4, f'static/images/{image_name}'))

    return jsonify({'predictions': predictions, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
