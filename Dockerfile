# Imagen base con TensorFlow (incluye Python)
FROM tensorflow/tensorflow:2.18.0

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY requirements.txt .
COPY serve.py .
COPY app ./app
COPY modelo_covid.h5 ./modelo_covid.h5

# Instala las dependencias
RUN pip install -r requirements.txt

# Crea la carpeta para las subidas
RUN mkdir -p /app/uploads

# Expone el puerto que usa Flask
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "serve.py"]
