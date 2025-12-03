import joblib
import numpy as np
import pandas as pd

# Cargar artefactos entrenados (una sola vez)
preprocesamiento = joblib.load("preprocesamiento.pkl")
preprocesamiento_class = joblib.load("preprocesamiento_class.pkl")
modelo_reg = joblib.load("model_reg.pkl")
modelo_clas = joblib.load("model_clas.pkl")


def predecir_precio(df: pd.DataFrame) -> float:
    """Recibe un DataFrame con UNA fila y las columnas del modelo de regresión
    y devuelve el precio predicho (float).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predecir_precio espera un DataFrame de pandas")
    if df.shape[0] != 1:
        raise ValueError("predecir_precio espera un DataFrame con exactamente 1 fila")

    # Transformar con el pipeline de preprocesamiento
    X_nuevo = preprocesamiento.transform(df)
    # El modelo fue entrenado sobre el log-precio
    pred_log = modelo_reg.predict(X_nuevo)
    # Volver a la escala original
    return float(np.exp(pred_log[0]))


def predecir_recomendable(df: pd.DataFrame, threshold: float = 0.5) -> str:
    """Recibe un DataFrame con UNA fila y las columnas del modelo de clasificación.
    Devuelve 'Recomendable' o 'No recomendable' según la probabilidad predicha.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predecir_recomendable espera un DataFrame de pandas")
    if df.shape[0] != 1:
        raise ValueError(
            "predecir_recomendable espera un DataFrame con exactamente 1 fila"
        )

    X_nuevo = preprocesamiento_class.transform(df)
    prob = modelo_clas.predict_proba(X_nuevo)[0, 1]

    return "Recomendable" if prob >= threshold else "No recomendable"
