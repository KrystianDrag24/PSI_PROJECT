from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import io
import logging

app = Flask(__name__)

# Konfiguracja logowania – plik w tym samym katalogu
logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Wczytanie modelu
model = load_model(os.path.join('model', 'best_model_finetuned.h5'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    score = None

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            prediction_result = "Nie wybrano pliku."
            logging.warning("Nie wybrano pliku.")
        else:
            try:
                file = request.files['file']
                img = Image.open(io.BytesIO(file.read())).resize((224, 224)).convert('RGB')
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]
                score = float(prediction)

                if prediction >= 0.75:
                    prediction_result = "Nowotwór złośliwy"
                elif prediction >= 0.5:
                    prediction_result = "Podejrzane zmiany (niepewność)"
                else:
                    prediction_result = "Zdrowe płuca"

                logging.info(f'Predykcja: {prediction_result}, Score: {score:.4f}')

            except Exception as e:
                prediction_result = f"Błąd podczas przetwarzania obrazu: {str(e)}"
                logging.error(f'Błąd: {str(e)}')

    return render_template('index.html', prediction=prediction_result, score=score)

@app.route('/logs/')
def view_logs():
    try:
        with open('logs/predictions.log', 'r') as f:
            log_content = f.read().replace('\n', '<br>')
        return f"<h2>Logi predykcji:</h2><p>{log_content}</p>"
    except Exception as e:
        return f"Błąd podczas otwierania logów: {str(e)}"

print("Serwer Flask działa pod adresem: http://127.0.0.1:5000/")
print("Serwer Flask działa pod adresem(logi): http://127.0.0.1:5000/logs/")

if __name__ == '__main__':
    app.run(debug=True)
