# Librerías estándar
from datetime import datetime
import re
import unicodedata

# Librerías de terceros
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser
from datetime import time

# Scikit-learn: procesamiento y modelos
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# CatBoost
from catboost import CatBoostRegressor


def extraer_temperaturas(col):
    temps_inicial = []
    temps_final = []

    for val in col:
        if pd.isna(val):
            temps_inicial.append(None)
            temps_final.append(None)
            continue

        val = str(val).strip()

        # Eliminar espacios, dejar solo números y símbolos relevantes
        val = re.sub(r'\s+', '', val)

        # Reemplazar cualquier tipo de separador por uno solo
        val = re.sub(r'[/\-]', '/', val)

        # Agregar / si está pegado (ej: 175°192° → 175°/192°)
        if re.match(r'\d+°\d+°', val):
            val = re.sub(r'(\d+°)(\d+°)', r'\1/\2', val)

        # Quitar los símbolos de grado
        val = val.replace('°', '')

        # Extraer los dos números
        match = re.match(r'(\d{2,3})/(\d{2,3})', val)
        if match:
            temps_inicial.append(int(match.group(1)))
            temps_final.append(int(match.group(2)))
        else:
            temps_inicial.append(None)
            temps_final.append(None)

    return pd.Series(temps_inicial), pd.Series(temps_final)


def tiempo_a_minutos(val):
    if isinstance(val, time):
        return val.hour + val.minute / 60  # hora = minutos, minuto = segundos
    elif isinstance(val, str):
        val = val.strip().replace('.', ':')  # Manejar valores tipo '10.15'
        try:
            h, m = map(int, val.split(':'))
            return h + m / 60
        except:
            return None
    return None

def limpiar_columna_fecha(df, col):
    def parse_fecha(val):
        if isinstance(val, pd.Timestamp) or isinstance(val, datetime):
            return val.date()
        if isinstance(val, str):
            val = val.strip().replace(" ", "").replace("Ene", "Jan").replace("Abril", "April")
            try:
                return parser.parse(val, dayfirst=True).date()
            except:
                return pd.NaT
        return pd.NaT

    df[col] = df[col].apply(parse_fecha)
    return df

def normalizar_lote(lote):
    if pd.isna(lote):
        return None
    # Quitamos espacios, convertimos a string y estandarizamos separadores
    lote = str(lote).strip().replace(" ", "")

    # Extraemos solo los números
    numeros = re.findall(r'\d+', lote)
    if len(numeros) >= 2:
        parte1 = numeros[0].zfill(2)  # Nos aseguramos de que el lote tenga 2 dígitos antes de -
        parte2 = ''.join(numeros[1:]).zfill(6)[:6]  # Juntamos el resto y aseguramos 6 dígitos
        return f"{parte1}-{parte2}"
    return lote


def limpiar_columna_humedad(df):
    def convertir_humedad(val):
        if isinstance(val, str):
            val = val.replace(",", ".").strip()
        try:
            return float(val)
        except:
            return pd.NA

    df['%H - %'] = df['%H - %'].apply(convertir_humedad)
    return df

def creacion_tipo_cafe(df):
    # Creamos el df_codigos para generar la columna Tipo de café en base al Lote
    df_codigos = pd.DataFrame({
        'Código': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                   '21', '22', '23'],
        'Tipo de café': ['Madre Laura Lavado', 'Madre Laura Natural', 'Madre Laura Descafeinado',
                         'Doña Rosalba', 'Doña Dolly', 'Don Rafael PBW', 'Don Felix',
                         'Gesha Villabernarda', 'Don Juan Tabi', 'Doña Liceth',
                         'Monteverde - Wush Wush', 'El Ocaso - Pink Bourbon', 'El Cedrela - Pink Bourbon',
                         'Don Reinaldo', 'Don Victor - Maragogipe', 'Gesha Natural Villabernarda',
                         'Familia Bedoya Castaño', 'Don Johan', 'Familia Gutierrez - Gesha',
                         'Familia Vergara - Bourbon Sidra', 'Don Victor - Red Bourbon',
                         'El Ocaso - Caturron Natural', 'Esteban Robledo']
    })

    # Hacer una copia para no modificar el DataFrame original
    df = df.copy()

    # Asegurarse de que la columna 'Lote' esté en string y sin nulos
    df['Lote'] = df['Lote'].astype(str).fillna('')

    # Extraer código de los dos primeros caracteres
    df['Código'] = df['Lote'].str[:2]

    # Unir con la tabla de códigos
    df = df.merge(df_codigos, on='Código', how='left')

    return df



def limpiar_columna_notas(df):

  df['Notas de catación'] = (
    df_trillado['Notas de catación']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace('nan', np.nan)  # reemplaza 'nan' string por np.nan real
  )

  # Diccionarios de palabras clave a buscar en las notas
  sabores_principales = {
      'sabor_chocolate': ['chocolate', 'cacao', 'nibs de cacao'],
      'sabor_cítrico': ['cítrica', 'limón', 'mandarina', 'naranja', 'limonaria', 'citric'],
      'sabor_caramelo': ['caramelo', 'dulce de leche', 'azúcar morena'],
      'sabor_frutas': ['fruta', 'fresa', 'fresas', 'frambuesa', 'frutas', 'moras', 'uva', 'manzana', 'durazno', 'maracuyá', 'sandía', 'cereza', 'ciruela', 'arándano', 'frambuesa'],
      'sabor_floral': ['jazmín', 'rosas', 'flor', 'lavanda', 'cedrón'],
      'sabor_panela': ['panela', 'melao'],
      'sabor_miel': ['miel', 'maple'],
  }

  tipos_cuerpo = {
      'cuerpo_ligero': ['ligero'],
      'cuerpo_medio': ['medio'],
      'cuerpo_completo': ['completo', 'cremoso', 'sedoso', 'aterciopelado'],
  }

  tipos_acidez = {
      'acidez_brillante': ['acidez brillante'],
      'acidez_media': ['acidez media'],
      'acidez_suave': ['acidez suave', 'jugosa'],
      'acidez_alta': ['acidez alta'],
      'acidez_baja': ['acidez baja'],
  }

  tipos_final = {
      'final_prolongado': ['final prolongado', 'prolongado'],
      'final_limpio': ['final limpio'],
      'final_corto': ['final corto'],
      'final_amargo': ['final amargo'],
      'final_dulce': ['final dulce']
  }

  # Función para detectar keywords
  def detectar_keywords(texto, keywords):
      texto = str(texto).lower()
      return {col: int(any(k in texto for k in ks)) for col, ks in keywords.items()}

  # Crear columnas en df_trillado1_limpio
  for idx, fila in df.iterrows():
      texto = fila['Notas de catación']
      resultado = {}
      for grupo in [sabores_principales, tipos_cuerpo, tipos_acidez, tipos_final]:
          resultado.update(detectar_keywords(texto, grupo))
      for col, val in resultado.items():
          df.at[idx, col] = val

  # Rellenar NaN por 0 en columnas recién creadas
  cols_nuevas = [col for grupo in [sabores_principales, tipos_cuerpo, tipos_acidez, tipos_final] for col in grupo.keys()]
  df[cols_nuevas] = df[cols_nuevas].fillna(0).astype(int)



def limpiar_columna_puntaje(df):

    df = df.dropna(subset=['Puntaje - N°'])

    def convertir_puntaje(val):
        if isinstance(val, str):
            val = val.replace(",", ".").strip()
        try:
            return float(val)
        except:
            return pd.NA

    df.loc[:, 'Puntaje - N°'] = df['Puntaje - N°'].apply(convertir_puntaje)
    return df

def limpiar_columna_cnc(df):
    df.loc[:,'Puntaje - C/NC'] = df['Puntaje - C/NC'].str.strip().str.upper()
    return df

def limpiar_columna_liberacion(df):
    df.loc[:, 'Liberación de lote - SI/NO'] = df['Liberación de lote - SI/NO'].str.strip().str.upper().replace({'SI': 'Sí'})
    return df

def limpiar_columna_responsable(df):
    df.loc[:,'Responsable'] = df['Responsable'].str.strip().str.upper()
    return df

def limpiar_df_trillado(df):
    df = limpiar_columna_fecha(df, 'Fecha')
    df['Lote'] = df['Lote'].apply(normalizar_lote)
    df = limpiar_columna_humedad(df)
    limpiar_columna_notas(df)
    df = limpiar_columna_puntaje(df)
    df = limpiar_columna_cnc(df)
    df = limpiar_columna_liberacion(df)
    df = limpiar_columna_responsable(df)
    df = creacion_tipo_cafe(df)
    df = df.drop(['Denominación/marca','%H - C/NC', 'Mallas - C/NC',
       'Verificación física café tostado', 'Notas de catación',
       'Puntaje - C/NC', 'Código'], axis=1)
    
    return df

def carga_datasets():
    na_vals = ['n/a', 'na', 'n.a.', 'none', '-', '', 'null', 'N/A', 'NA', 'None', 'N/A ']
    # 2. Creamos los nombres de las columnas
    column_names = [
        "Fecha",
        "Lote",
        "Denominación/marca",
        "Cantidad",
        "%H - %",
        "%H - C/NC",
        "Mallas - #",
        "Mallas - C/NC",
        "Verificación física café tostado",
        "Notas de catación",
        "Puntaje - N°",
        "Puntaje - C/NC",
        "Liberación de lote - SI/NO",
        "Responsable"
    ]

    # 3. Leemos los datos desde la fila 9 (índice 8)
    dfs = pd.read_excel(
        "CC FT 17 Formato de Control de Calidad Café de Trillado.xlsx",
        skiprows=8,
        header=None,
        names=column_names,
        nrows=76,
        na_values=na_vals,
        sheet_name = [0,1]
    )
    df_trillado = pd.concat([dfs[0], dfs[1]], ignore_index=True)
    df_trillado = df_trillado.drop(df_trillado.index[-12:])

    dfs2 = pd.read_excel('CC FT 18 Formato de Tostión.xlsx', nrows=499, na_values=na_vals, header=5, sheet_name=[0,1])
    df_tostion= pd.concat([dfs2[0], dfs2[1]], ignore_index=True)
    return df_trillado,  df_tostion

df_trillado, df_tostion = carga_datasets()
df_trillado_limpio = limpiar_df_trillado(df_trillado)

df_tostion['Tiempo_tueste_min'] = df_tostion['Tiempo de tueste'].apply(tiempo_a_minutos)



df_tostion.columns = df_tostion.columns.str.rstrip()

df_tostion = limpiar_columna_fecha(df_tostion, 'Fecha')

df_tostion['Lote'] = df_tostion['Lote'].apply(normalizar_lote)

df_tostion['Origen'] = df_tostion['Origen'].str.strip().replace({
    'Herrra': 'Herrera',
    'Ciudad Bolivar': 'Ciudad Bolívar',  # Normalizar tilde si aplica
})

df_tostion['Variedad'] = (
    df_tostion['Variedad']
    .str.strip()
    .replace({
        'Caturron': 'Caturra',
        'Red Bourbon': 'Bourbon Rojo',
    })
)
df_tostion['Beneficio'] = df_tostion['Beneficio'].str.strip().str.capitalize()
df_tostion['Perfil'] = (
    df_tostion['Perfil']
    .str.strip()
    .replace({
        'Filtrados': 'Filtrado',
        'Espressso': 'Espresso'
    })
)
df_tostion['Proceso'] = df_tostion['Proceso'].str.strip().str.capitalize()
df_tostion["TempInicio"], df_tostion["TempFinal"] = extraer_temperaturas(df_tostion["Temp. De inicio y final"])
df_tostion['Tostador'] = df_tostion['Tostador'].str.replace(' ', '').str.upper()
df_tostion = creacion_tipo_cafe(df_tostion)
df_tostion.drop(columns=['Temp. De inicio y final', 'Tiempo de tueste', 'Observaciones', 'Código'], inplace=True)

# Calcular el Origen más frecuente por cada Lote
origen_mas_frecuente = df_tostion.groupby('Lote')['Origen'].agg(lambda x: x.mode().iloc[0]).reset_index()

# Hacer merge con df_trillado_limpio
df_nuevo = df_trillado_limpio.merge(origen_mas_frecuente, on='Lote', how='left')

df_nuevo # Vemos y eliminemos correlaciones
df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'], errors='coerce')
df_nuevo['Año'] = df_nuevo['Fecha'].dt.year
df_nuevo['Mes'] = df_nuevo['Fecha'].dt.month
df_nuevo['Día'] = df_nuevo['Fecha'].dt.day
df_nuevo = df_nuevo.drop(columns=['Fecha'])

# Lista de columnas a conservar (además de la objetivo)
columnas_a_conservar = [
    'Origen',             # Mejor que Tipo de Café  
    'sabor_caramelo',     # Representante de sabores
    'cuerpo_medio',       # Representante de cuerpo
    'acidez_media',       # Representante de acidez
    'final_dulce',        # Representante de final
    'Puntaje - N°'        # Variable objetivo (no se elimina)
]

# Creamos df_nuevo solo con esas columnas
df_nuevo = df_nuevo[[col for col in columnas_a_conservar if col in df_nuevo.columns]]

print(df_trillado_limpio.columns)





print('-------------Entrenamiento GradientBoostingRegressor-------------')
# Copia del DataFrame
df_model = df_nuevo.copy()

# Elimina filas con valores faltantes en la variable objetivo
df_model = df_model.dropna(subset=['Puntaje - N°'])

# Extrae X e y
X = df_model.drop(columns=['Puntaje - N°'])  # puedes eliminar 'Lote' si no lo vas a usar
y = df_model['Puntaje - N°']

# Columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Fecha' in categorical_cols:
    categorical_cols.remove('Fecha')  # será tratada aparte

# Preprocesamiento completo
preprocessor_gb = Pipeline(steps=[
    ('encoder', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough'))
])

# Pipeline final
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor_gb),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'mse': make_scorer(mean_squared_error),
    'r2': make_scorer(r2_score)
}

results = cross_validate(pipeline_gb, X, y, cv=cv, scoring=scoring, return_train_score=False)
print('------Gradient con validación cruzada (KFold con 5 particiones)-----')
# Resultados
print("MSE promedio:", results['test_mse'].mean())
print("MSE std:", results['test_mse'].std())
print("R2 promedio:", results['test_r2'].mean())
print("R2 std:", results['test_r2'].std())



# Obtiene predicciones de validación cruzada (nunca ve los datos de test)
y_pred = cross_val_predict(pipeline_gb, X, y, cv=cv)

print('-----cross_val_predict con el Pipeline anterior-----')
# Métricas globales
print("MSE total:", mean_squared_error(y, y_pred))
print("R2 total:", r2_score(y, y_pred))

# Aplica preprocesamiento por separado
X_processed = preprocessor_gb.fit_transform(X)

# Si es sparse, lo pasamos a denso (CatBoost no admite sparse)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Entrenamos el modelo CatBoost directamente
model = GradientBoostingRegressor(verbose=0, random_state=42)
model.fit(X_processed, y)

# Obtén nombres de las features transformadas
feature_names = preprocessor_gb.named_steps['encoder'].get_feature_names_out()

# Obtén importancias
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Visualiza
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15][::-1], importance_df['Importance'][:15][::-1])
plt.xlabel('Importancia')
plt.title('Top 15 Features más importantes (GradientBoost)')
plt.tight_layout()
plt.show()


print('------CatBoostRegressor con validación cruzada (KFold con 5 particiones)-----')


# Copia del DataFrame
df_model = df_nuevo.copy()

# Elimina filas con valores faltantes en la variable objetivo
df_model = df_model.dropna(subset=['Puntaje - N°'])

# Extrae X e y
X = df_model.drop(columns=['Puntaje - N°'])  # puedes eliminar 'Lote' si no lo vas a usar
y = df_model['Puntaje - N°']

# Columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocesamiento completo
preprocessor_ct = Pipeline(steps=[
    ('encoder', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough'))
])


# Pipeline final con LGBMRegressor
pipeline_ct = Pipeline(steps=[
    ('preprocessor', preprocessor_ct),
    ('model', CatBoostRegressor(verbose=0, random_state=42))
])

# Validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'mse': make_scorer(mean_squared_error),
    'r2': make_scorer(r2_score)
}

results = cross_validate(pipeline_ct, X, y, cv=cv, scoring=scoring, return_train_score=False)


# Resultados
print("MSE promedio:", results['test_mse'].mean())
print("MSE std:", results['test_mse'].std())
print("R2 promedio:", results['test_r2'].mean())
print("R2 std:", results['test_r2'].std())

print('-----cross_val_predict con el Pipeline anterior-----')
# Obtiene predicciones de validación cruzada (nunca ve los datos de test)
y_pred_ct = cross_val_predict(pipeline_ct, X, y, cv=cv)

# Métricas globales
print("MSE total:", mean_squared_error(y, y_pred_ct))
print("R2 total:", r2_score(y, y_pred))



# Aplica preprocesamiento por separado
X_processed = preprocessor_ct.fit_transform(X)

# Si es sparse, pásalo a denso (CatBoost no admite sparse)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Entrena el modelo CatBoost directamente
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_processed, y)

# Obtén nombres de las features transformadas
feature_names = preprocessor_ct.named_steps['encoder'].get_feature_names_out()

# Obtén importancias
importances = model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Visualiza
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15][::-1], importance_df['Importance'][:15][::-1])
plt.xlabel('Importancia')
plt.title('Top 15 Features más importantes (CatBoost)')
plt.tight_layout()
plt.show()




print('------RandomForestRegressor con validación cruzada (KFold con 5 particiones)-----')


df_train = df_nuevo.copy()

X = df_train.drop(columns=['Puntaje - N°'])
y = df_train['Puntaje - N°']

# Identificamos las columnas categóricas y numéricas
cat_features = ['Origen']  # ajusta si hay más
num_features = [col for col in X.columns if col not in cat_features]

# Preprocesamiento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # imputa valores faltantes numéricos
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# Pipeline completo con RandomForest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print(f"MSE promedio: {-mse_scores.mean():.4f}")
print(f"MSE std: {mse_scores.std():.4f}")
print(f"R2 promedio: {r2_scores.mean():.4f}")
print(f"R2 std: {r2_scores.std():.4f}")

print('-----cross_val_predict con el Pipeline anterior-----')

y_pred = cross_val_predict(model, X, y, cv=cv)

# Métricas globales
print("MSE total:", mean_squared_error(y, y_pred))
print("R2 total:", r2_score(y, y_pred))

# Entrena el modelo primero
model.fit(X, y)

# Accede al OneHotEncoder
onehot_encoder = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
encoded_cols = onehot_encoder.get_feature_names_out(['Origen'])

# Obtiene todos los nombres de columnas finales (numéricas + codificadas)
final_feature_names = num_features + list(encoded_cols)

# Importancia de características
importances = model.named_steps['regressor'].feature_importances_


# Muestra en DataFrame ordenado
feature_imp_df = pd.DataFrame({
    'Feature': final_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_imp_df)
# Graficar
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title('Importancia de las características para RandomForest')
plt.tight_layout()
plt.show()