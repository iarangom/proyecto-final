#APP DE PABLO 


from modelo_reg_funcion import predecir_precio #Esto importa la funcion creada en el archivo modelo reg. Para predecir el precio solo tienes que llamar la funcion 

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