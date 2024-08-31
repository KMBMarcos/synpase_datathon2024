import pandas as pd
import numpy as np

# Cargar el DataFrame
consumos_df = pd.read_csv('./data/df_entrenamiento/consumos_participantes.csv')

# Función para verificar si la moneda corresponde al país según las nuevas reglas
def verificar_moneda(row):
    pais = row['codigo_pais_comercio']
    moneda = row['moneda_liquidacion']
    
    if pais == 'DOM':
        return moneda == 'DOP'
    elif pais == 'USA':
        return moneda == 'USD'
    else:
        return moneda == 'USD'  # Para todos los demás países, incluyendo NLD

# Crear una nueva columna que indique si la moneda es la correspondiente
consumos_df['moneda_correcta'] = consumos_df.apply(verificar_moneda, axis=1)

# Convertir valores True/False a 1/0
consumos_df['moneda_correcta'] = consumos_df['moneda_correcta'].astype(int)

# Mostrar las primeras filas del DataFrame modificado
# print(consumos_df[['codigo_pais_comercio', 'moneda_liquidacion', 'moneda_correcta']].head())


# Seleccionar los países de interés
paises_interes = ['DOM', 'USA', 'NLD']

# Crear una nueva columna para agrupar los demás países bajo "Other"
consumos_df['codigo_pais_comercio_grouped'] = consumos_df['codigo_pais_comercio'].apply(
    lambda x: x if x in paises_interes else 'Other'
)

# Aplicar One-Hot Encoding
one_hot_encoded = pd.get_dummies(consumos_df['codigo_pais_comercio_grouped'], prefix='pais')

# Concatenar el resultado con el DataFrame original
consumos_df = pd.concat([consumos_df, one_hot_encoded], axis=1)

# Mostrar las primeras filas del DataFrame modificado
print(consumos_df.head())