import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd

# 1. Cargar preprocesador y modelo
preprocesamiento = joblib.load("preprocesamiento_modelo.pkl")
mode_reg = mlflow.sklearn.load_model("model_reg.pkl")

# 2. DataFrame de entrada con las MISMAS columnas que X_train original
df_entrada = pd.DataFrame([{
    "bathroomsf": , # 
    'bedrooms': , # Numero de cuartos
    'room_type': ,
    'review_scores_location': ,
    'has_jacuzzi': ,
    'has_tv_cable': ,
    'has_pool': ,
    'estimated_occupancy_l365d': ,
    'review_scores_rating': ,
    'accommodates': ,
}])

# 3. Normalizar / transformar con el mismo preprocesamiento
X_nuevo_proc = preprocesamiento.transform(df_entrada)

# 4. Predecir (recuerda: el modelo se entrenó con log(y))
pred_log = modelo_10_17.predict(X_nuevo_proc)
pred_price = np.exp(pred_log)

print("Precio predicho:", float(pred_price[0]))
