import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd

# Solo 1 vez es necesario 
preprocesamiento = joblib.load("preprocesamiento.pkl")
modelo_reg = joblib.load("model_reg.pkl")
modelo_clas = joblib.load("model_clas.pkl")


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

df_entrada = pd.DataFrame([{
    'host_response_rate': 0.95,     # decimal entre 0 y 1
    'host_is_superhost': 1,         # 1 si es superhost, 0 si no
    'host_identity_verified': 1,    # 1 si verificó identidad, 0 si no
    'instant_bookable': 1,          # 1 si se puede reservar instantáneamente
    'minimum_nights': 2,            # noches mínimas
    'maximum_nights': 30,           # noches máximas
    'bathroomsf': 1.5,              # cantidad de baños (decimal o entero)
    'beds': 2,                      # número de camas
    'has_wifi': 1,                  # amenity (1/0)
    'has_ac_heating': 0,            # amenity (1/0)
    'has_self_checkin': 1,          # amenity (1/0)
    'has_tv_cable': 1               # amenity (1/0)
}])

def predecir_recomendable(df_entrada, threshold=0.5):
    """
    df_entrada: DataFrame con 1 fila y las columnas empleadas en el modelo.
    threshold: valor de corte para clasificar (default 0.5).

    Returns:
        recomendacion (str): "Recomendable" o "No recomendable"
    """

    # Preprocesar
    X_nuevo = preprocesamiento.transform(df_entrada)
    # Probabilidad predicha de clase 1 (recomendable)
    prob = modelo_clas.predict_proba(X_nuevo)[0, 1]
    # Predicción textual
    if prob >= threshold:
        return "Recomendable"
    else:
        return "No recomendable"
    
print(predecir_recomendable(df_entrada))