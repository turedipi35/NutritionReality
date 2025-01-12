from flask import Flask, request
import os
import torch
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Ricevi i dati binari inviati
        file_data = request.data
        
        if not file_data:
            return 'No data received', 400

        # Crea un nome per il file
        filename = os.path.join(UPLOAD_FOLDER, f"frame_{len(os.listdir(UPLOAD_FOLDER)) + 1}.png")

        # Salva il file
        with open(filename, 'wb') as f:
            f.write(file_data)

        print(f"File salvato: {filename}")  # Log per confermare che il file è stato salvato
        # Esegui YOLO sull'immagine caricata
        results = model(Image.open(filename))

        # Ottieni i risultati come dizionario
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'detections': detections
        }), 200

    except Exception as e:
            print(f"Error during upload: {e}")
            return jsonify({'error': 'Error during upload', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'private_key.pem'))
