import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd

# 1. Cargar preprocesador y modelo
preprocesamiento = joblib.load("preprocesamiento.pkl")
modelo_reg = joblib.load("model_reg.pkl")

# 2. DataFrame de entrada con las MISMAS columnas que X_train original
df_entrada = pd.DataFrame([{
    "bathroomsf":1.5 ,                  # Numero de baños (puede ser decimal)
    'bedrooms':2 ,                      # Numero de cuartos (puede ser decimal)
    'room_type':1,                      # Binario 'Entire home/apt': 1, 'Private room': 0
    'review_scores_location':4.1 ,      # Decimal [0,5]
    'has_jacuzzi':0 ,                   # Binario : 1 si si tiene, 0 si no
    'has_tv_cable':1 ,                  # Binario : 1 si si tiene, 0 si no
    'has_pool':0 ,                      # Binario : 1 si si tiene, 0 si no
    'estimated_occupancy_l365d': 234 ,  #Numero entero [0,365]
    'review_scores_rating': 4.6 ,       # Decimal [0,5]
    'accommodates': 4 ,                 # Entero numero de personas que pueden dormir ahi
}])

# Definir que variables de entrada del df son numericas. Necesario para procesar adecuadamente
x_num= ["bathroomsf","bedrooms",'review_scores_location','estimated_occupancy_l365d','review_scores_rating','accommodates']


# 3. Normalizar / transformar con el mismo preprocesamiento
X_nuevo_proc = preprocesamiento.transform(df_entrada)

# 4. Predecir (recuerda: el modelo se entrenó con log(y))
pred_log = modelo_reg.predict(X_nuevo_proc)
# RESULTADO DE MODELO
pred_price = np.exp(pred_log)

print("Precio predicho:", float(pred_price[0]))
