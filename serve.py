from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from app.model import load_covid_model
from app.predict import preprocess_image, predict_covid

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'modelo_covid.h5'
STATIC_FOLDER = 'static'  # Nueva carpeta para archivos estáticos

# Crear carpetas necesarias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Cargar el modelo al iniciar
model = load_covid_model(MODEL_PATH)

# Ruta para la página principal
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>COVID-19 Detector</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 15px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>COVID-19 Detector</h1>
        <p>Suba una imagen de rayos X de tórax para análisis:</p>
        <input type="file" id="xray" accept="image/*">
        <button onclick="predict()">Analizar</button>
        <div id="result"></div>

        <script>
            async function predict() {
                const file = document.getElementById('xray').files[0];
                if (!file) {
                    document.getElementById('result').innerHTML = '<p style="color:red;">Por favor seleccione un archivo</p>';
                    return;
                }
                
                const form = new FormData();
                form.append('file', file);
                
                document.getElementById('result').innerHTML = '<p>Analizando imagen...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: form
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        document.getElementById('result').innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
                    } else {
                        const confidencePercent = (data.confidence * 100).toFixed(2);
                        document.getElementById('result').innerHTML = `
                            <h3>Resultado: ${data.class}</h3>
                            <p>Confianza: ${confidencePercent}%</p>
                            ${data.class === 'COVID-19' ? 
                              '<p style="color:red;">¡Se detectó COVID-19! Consulte a un médico.</p>' : 
                              '<p style="color:green;">No se detectó COVID-19.</p>'}
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            img_array = preprocess_image(filepath)
            prediction = predict_covid(model, img_array)
            os.remove(filepath)
            
            # Formatea la respuesta para la interfaz
            return jsonify({
                'class': 'No COVID-19' if prediction['prediction'] > 0.5 else 'COVID-19',
                'confidence': prediction['confidence'],
                'probability': prediction['prediction']
            })
        except Exception as e:
            return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido. Use JPG o PNG'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)