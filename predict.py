import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv


load_dotenv()

DATASET = os.getenv("DATASET")
TARGET = os.getenv("TARGET")
MODEL_NAME = os.getenv("MODEL")
TRIALS = int(os.getenv("TRIALS"))
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE")
INPUT_FOLDER = os.getenv("INPUT_FOLDER")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
PORT = int(os.getenv("PORT", 8000))

MODELS = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}


def preprocess_data(data, target):
    X = data.drop(columns=[target])
    y = data[target]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def optimize_model(model, X, y):
    param_grid = {
        "RandomForest": {'n_estimators': [50, 100], 'max_depth': [5, 10]},
        "GradientBoosting": {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
        "SVM": {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
        "KNN": {'n_neighbors': [3, 5, 7]},
        "NaiveBayes": {}
    }
    grid_search = GridSearchCV(model, param_grid[MODEL_NAME], cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def train_model():
    data = pd.read_parquet(DATASET)
    X, y, preprocessor = preprocess_data(data, TARGET)
    model = MODELS[MODEL_NAME]

    best_model = optimize_model(model, X, y) if TRIALS > 1 else model
    best_model.fit(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"Precisión: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print("Matriz de confusión:")
    print(conf_matrix)

    joblib.dump(best_model, "model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")
    print(
        f"Modelo {MODEL_NAME} entrenado y guardado con éxito.")


def batch_prediction():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Creando la carpeta de entrada: {INPUT_FOLDER}")
        os.makedirs(INPUT_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        print(f"Creando la carpeta de salida: {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".parquet"):
            data = pd.read_parquet(os.path.join(INPUT_FOLDER, filename))
            X = preprocess_data(data, TARGET)[0]
            model = joblib.load("model.joblib")
            predictions = model.predict_proba(X)
            output_df = pd.DataFrame(predictions, columns=[f"Clase_{i + 1}" for i in range(predictions.shape[1])])
            output_path = os.path.join(OUTPUT_FOLDER, f"{filename}_predictions.parquet")
            output_df.to_parquet(output_path)
            print(
                f"Predicciones guardadas en {output_path}")


app = FastAPI()


class PredictionInput(BaseModel):
    samples: List[Dict]


@app.post("/predict")
async def predict(input_data: PredictionInput):
    model = joblib.load("model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    samples = pd.DataFrame(input_data.samples)
    X = preprocessor.transform(samples)
    predictions = model.predict_proba(X)
    class_names = [f"Clase_{i + 1}" for i in range(predictions.shape[1])]
    output = [{"Clase": dict(zip(class_names, pred))} for pred in predictions]
    return {"predictions": output}


if __name__ == '__main__':
    train_model()
    if DEPLOYMENT_TYPE == "Batch":
        batch_prediction()
    elif DEPLOYMENT_TYPE == "API":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=PORT)
    else:
        print("DEPLOYMENT_TYPE no válido.")
