import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd

# Solo 1 vez es necesario 
preprocesamiento = joblib.load("preprocesamiento.pkl")
modelo_reg = joblib.load("model_reg.pkl")

#FUNCION PARA PREDECIR EL PRECIO CON EL MODELO

def predecir_precio(datos_dict):
    """
    datos_dict: diccionario con las 11 keys de FEATURES.
    Devuelve el precio predicho (float).
    """
    df_entrada = pd.DataFrame([datos_dict], columns=FEATURES)
    X_nuevo = preprocesamiento.transform(df_entrada)
    pred_log = model_reg.predict(X_nuevo)
    return float(np.exp(pred_log)[0])

#EJEMPLO DE DF QUE USAR LA FUNCION predecir_precio
df_entrada = pd.DataFrame([{
    "bathroomsf":2 ,                  # Numero de baños (puede ser decimal)
    'bedrooms':4,                      # Numero de cuartos (puede ser decimal)
    'room_type':1,                      # Binario 'Entire home/apt': 1, 'Private room': 0
    'review_scores_location':4.6 ,      # Decimal [0,5]
    'has_jacuzzi':1 ,                   # Binario : 1 si si tiene, 0 si no
    'has_tv_cable':1 ,                  # Binario : 1 si si tiene, 0 si no
    'has_pool':0 ,                      # Binario : 1 si si tiene, 0 si no
    'estimated_occupancy_l365d': 234 ,  #Numero entero [0,365]
    'review_scores_rating': 4.6 ,       # Decimal [0,5]
    'accommodates': 4 ,                 # Entero numero de personas que pueden dormir ahi
    }])

    