# **Instrucciones para Reproducir el Proyecto**

## 1. **Preparación del Entorno**

**Archivos Necesarios**:
   - `predict.py`: Código principal del proyecto.
   - `Dockerfile`: Archivo para construir la imagen Docker.
   - `requirements.txt`: Lista de dependencias necesarias.
   - `.env`: Archivo de configuración.
   - **Directorio de Datos**:
     - Crea las carpetas:
       ```bash
       mkdir -p data/input data/output
       ```
     - Coloca los archivos Parquet en `data/input`.

## 2. **Configura el Archivo `.env`**:
   Ajusta el archivo `.env` según el modo de producción que desees utilizar:

   ### Modo Batch Prediction:
   ```env
   DATASET=/data/input/dataset.parquet
   TARGET=label_column
   MODEL=RandomForest
   TRIALS=10
   DEPLOYMENT_TYPE=Batch
   INPUT_FOLDER=/data/input
   OUTPUT_FOLDER=/data/output
   ```

  ### Modo API:
  ```env
  DATASET=/data/dataset.parquet
  TARGET=label_column
  MODEL=GradientBoosting
  TRIALS=15
  DEPLOYMENT_TYPE=API
  PORT=8000
  ```
  
## 3. **Costruccion del Contenedor**

```bash
    docker build -t automl-dockerizer:latest .
```

## 4. **Levantar la imagen de Docker**

  ### Modo Batch:
  Asegúrate de que el archivo .env esté configurado con DEPLOYMENT_TYPE=Batch.

  ```bash
    docker run --env-file .env -v $(pwd)/data:/data automl-dockerizer:latest
  ```

  ### Modo API:

  Asegúrate de que el archivo .env esté configurado con DEPLOYMENT_TYPE=API.

  ```bash
    docker run --env-file .env -p 8000:8000 -v $(pwd)/data:/data automl-dockerizer:latest
  ```

## 5. **Pruebas**

  Batch Prediction:
  1. Coloca un archivo Parquet en data/input.
  2. Ejecuta el contenedor.
  3. Los resultados aparecerán en data/output.
    API:
  
  Ve a http://localhost:8000/docs para acceder a la documentación interactiva.

Envía datos en formato JSON al endpoint /predict. Por ejemplo:

  ```bash
  {
    "samples": [
        {"feature1": 3, "feature2": "A", "area": 3000, "bathrooms": 2}
    ]
  }
  ```
Puedes usar herramientas como Postman o curl:

```bash

```
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "samples": [
               {"feature1": 3, "feature2": "A", "area": 3000, "bathrooms": 2}
           ]
         }'




