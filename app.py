import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer

# ==============================
# 🎨 CONFIGURACIÓN GLOBAL DE ESTILO
# ==============================
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['figure.facecolor'] = '#1e1e1e'   # Fondo de figura
plt.rcParams['axes.facecolor'] = '#1e1e1e'     # Fondo del área de gráficos

sns.set_style("darkgrid", {'axes.facecolor': '#1e1e1e'})

# ==============================
# 📥 CARGA DE DATOS
# ==============================
df = pd.read_csv("data.csv")  # Cambia por tu dataset real

# ==============================
# 🧼 IMPUTACIÓN DE VALORES FALTANTES
# ==============================
# Ejemplo: imputar valores numéricos con la media
imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=np.number).columns] = imputer.fit_transform(df.select_dtypes(include=np.number))

# ==============================
# 📊 ANÁLISIS DE VALORES FALTANTES
# ==============================
faltantes = df.isnull().sum()
faltantes = faltantes[faltantes > 0].sort_values(ascending=False)

# Bloque visual estilizado
st.markdown("""
    <div style="
        background-color: #2c2c2c;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.5;
        border: 1px solid #555;
    ">
    <strong>⚠️ Análisis:</strong> Las variables CO y NO2 concentran la mayoría de valores faltantes,
    lo que podría sesgar el análisis si no se trata adecuadamente. Se aplicó imputación por media
    para minimizar este efecto y conservar la estructura de los datos.
    </div>
""", unsafe_allow_html=True)

# ==============================
# 📉 GRAFICA DE FALTANTES
# ==============================
if not faltantes.empty:
    plt.figure(figsize=(10, 6))
    faltantes.plot(kind='bar')
    plt.title('Valores Faltantes por Variable')
    plt.xlabel('Variables')
    plt.ylabel('Cantidad de Valores Faltantes')
    st.pyplot(plt)

# ==============================
# 📈 MATRIZ DE CORRELACIÓN
# ==============================
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
st.pyplot(plt)

# ==============================
# 📊 ANÁLISIS BIVARIADO DE EJEMPLO
# ==============================
# (Reemplaza 'variable1' y 'variable2' con tus variables reales)
if 'variable1' in df.columns and 'variable2' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['variable1'], y=df['variable2'])
    plt.title('Análisis Bivariado: Variable 1 vs Variable 2')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    st.pyplot(plt)

# ==============================
# 📋 DATOS FINALES
# ==============================
st.write("Vista previa de los datos procesados:")
st.dataframe(df.head())
