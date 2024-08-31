import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the DataFrame
quejas_df = pd.read_csv('./data/df_entrenamiento/df_quejas_participantes.csv')

# Handle missing data
# quejas_df['cerrado_ticket'].fillna(quejas_df['cerrado_ticket'].mode()[0], inplace=True)
quejas_df['escalado_ticket'].fillna(False, inplace=True)
 
 
# Convert True/False to 1/0
quejas_df['cerrado_ticket'] = quejas_df['cerrado_ticket'].astype(int)
quejas_df['escalado_ticket'] = quejas_df['escalado_ticket'].astype(int)

# Normalize the text in 'canal_reclamacion_regulador' and other string columns
quejas_df['canal_reclamacion_regulador'] = quejas_df['canal_reclamacion_regulador'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
quejas_df['tipo_registro'] = quejas_df['tipo_registro'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
quejas_df['producto_aprobado'] = quejas_df['producto_aprobado'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# One-Hot Encoding for categorical variables
quejas_df = pd.get_dummies(quejas_df, columns=['canal_reclamacion_regulador', 'tipo_registro', 'producto_aprobado'])

# Convert dates to a consistent format and create a duration feature
quejas_df['fecha_apertura_reclamacion'] = pd.to_datetime(quejas_df['fecha_apertura_reclamacion'], errors='coerce')
quejas_df['fecha_resolucion_reclamacion'] = pd.to_datetime(quejas_df['fecha_resolucion_reclamacion'], errors='coerce')

# Create a duration feature
quejas_df['duracion_resolucion'] = (quejas_df['fecha_resolucion_reclamacion'] - quejas_df['fecha_apertura_reclamacion']).dt.days

# Drop original date columns if they are no longer needed
quejas_df.drop(['fecha_apertura_reclamacion', 'fecha_resolucion_reclamacion'], axis=1, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
quejas_df[['duracion_resolucion', 'numero_fila_activa', 'numero_fila_cerradas']] = scaler.fit_transform(
    quejas_df[['duracion_resolucion', 'numero_fila_activa', 'numero_fila_cerradas']])

# The DataFrame `quejas_df` is now ready for use in a machine learning model.
print(quejas_df.head())
print(quejas_df.info())

print('hola')