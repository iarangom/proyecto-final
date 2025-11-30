#APP DE PABLO 
import pandas as pd

from modelo_reg_funcion import predecir_precio #Esto importa la funcion creada en el archivo modelo reg. Para predecir el precio solo tienes que llamar la funcion 
from modelo_reg_funcion import predecir_recomendable
'''EJEMPLO DEL PARAMETRO ESPERADO POR predecir_precio()
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

    SE LLAMA ASI ---> predecir_precio( df_entrada)
'''

'''
PARA QUE EL MODELO CORRAR BIEN HAY QUE CREAR UN ENTORNO EN LA MAQUINA VIRTUAL EN LA CARPETA Modelo regresion.copy
# 2. Crear el entorno con las mismas versiones que usé yo
conda env create -f conda.yaml -n modelo_regresion
# 3. Activar el entorno
conda activate modelo_regresion
'''



#EJEMPLO DE DF QUE RECIBE CADA FUNCION #BORRALOS DESPUES

#DF para predecir precios
df_entrada1 = pd.DataFrame([{
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
#DF para predecir recomendable o no 
df_entrada2 = pd.DataFrame([{
    'host_response_rate': 0.05,     # decimal entre 0 y 1
    'host_is_superhost': 0,         # 1 si es superhost, 0 si no
    'host_identity_verified': 1,    # 1 si verificó identidad, 0 si no
    'instant_bookable': 0,          # 1 si se puede reservar instantáneamente
    'minimum_nights': 2,            # noches mínimas
    'maximum_nights': 4,           # noches máximas
    'bathroomsf': 1.5,              # cantidad de baños (decimal o entero)
    'beds': 2,                      # número de camas
    'has_wifi': 1,                  # amenity (1/0)
    'has_ac_heating': 0,            # amenity (1/0)
    'has_self_checkin': 0,          # amenity (1/0)
    'has_tv_cable': 0               # amenity (1/0)
}])

print(predecir_precio(df_entrada1)) #EJEMPLO DE COMO LLAMAR A LAS FUNCIONES BORRAR
print(predecir_recomendable(df_entrada2)) #EJEMPLO DE COMO LLAMAR A LAS FUNCIONES BORRAR