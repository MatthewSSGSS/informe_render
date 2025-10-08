import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(page_title="Análisis de Calidad del Aire - Beijing", page_icon="🌫️", layout="wide")

# ==============================
# CSS PERSONALIZADO (letras blancas)
# ==============================
st.markdown("""
    <style>
    body, .stMarkdown, .stText, div, p {
        color: white !important;
    }
    .block-container {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e222b;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# CARGA DE DATOS
# ==============================
@st.cache_data
def cargar_datos():
    df = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228.csv")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No"])
    return df

df = cargar_datos()

# ==============================
# VARIABLES DISPONIBLES
# ==============================
contaminantes = ["PM2.5"]
variables_meteo = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Opciones de Análisis")
var1 = st.sidebar.selectbox("Variable 1", contaminantes + variables_meteo)
var2 = st.sidebar.selectbox("Variable 2 (para análisis bivariado)", contaminantes + variables_meteo)

# ==============================
# PESTAÑAS
# ==============================
tabs = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Distribuciones",
    "🔗 Correlaciones",
    "📅 Estacionalidad",
    "🚨 Datos Faltantes",
    "🔀 Análisis Bivariado",
    "📝 Conclusiones"
])

# ==============================
# RESUMEN EJECUTIVO
# ==============================
with tabs[0]:
    st.subheader("📊 Resumen Ejecutivo")

    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio PM2.5", f"{df['PM2.5'].mean():.2f} µg/m³")
    col2.metric("Promedio TEMP", f"{df['TEMP'].mean():.2f} °C")
    col3.metric("Promedio PRES", f"{df['PRES'].mean():.2f} hPa")

    df_mensual = df.groupby(df["datetime"].dt.to_period("M")).mean(numeric_only=True)
    df_mensual.index = df_mensual.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_mensual.index, df_mensual["PM2.5"], label="PM2.5", color="orange")
    ax.set_title("Evolución mensual del PM2.5", color="white")
    ax.legend()
    st.pyplot(fig)

# ==============================
# DISTRIBUCIONES
# ==============================
with tabs[1]:
    st.subheader("📈 Distribuciones de Variables")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[var1], kde=True, color="orange", ax=ax)
        ax.set_title(f"Distribución de {var1}", color="white")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df[var2], kde=True, color="skyblue", ax=ax)
        ax.set_title(f"Distribución de {var2}", color="white")
        st.pyplot(fig)

# ==============================
# CORRELACIONES
# ==============================
with tabs[2]:
    st.subheader("🔗 Matriz de Correlación")

    corr = df[["PM2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matriz de Correlación", color="white")
    st.pyplot(fig)

# ==============================
# ESTACIONALIDAD
# ==============================
with tabs[3]:
    st.subheader("📅 Estacionalidad Mensual")

    df["month"] = df["datetime"].dt.month
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="month", y=var1, color="orange", ax=ax)
    ax.set_title(f"Estacionalidad de {var1} por mes", color="white")
    st.pyplot(fig)

# ==============================
# DATOS FALTANTES
# ==============================
with tabs[4]:
    st.subheader("🚨 Datos Faltantes")

    st.write("**Antes de Imputación:**")
    faltantes_inicial = df.isna().mean() * 100
    st.bar_chart(faltantes_inicial)

    # Tipo de ausencia (simple)
    st.markdown("**Tipo de Ausencia:** Se asume MCAR (Missing Completely At Random).")

    # Imputación
    imputer = SimpleImputer(strategy="mean")
    df_imputado = df.copy()
    df_imputado[["PM2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]] = imputer.fit_transform(
        df[["PM2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]]
    )

    st.write("**Después de Imputación:**")
    faltantes_final = df_imputado.isna().mean() * 100
    st.bar_chart(faltantes_final)

# ==============================
# ANÁLISIS BIVARIADO
# ==============================
with tabs[5]:
    st.subheader("🔀 Análisis Bivariado")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=var1, y=var2, alpha=0.5, color="orange")
    ax.set_title(f"Relación entre {var1} y {var2}", color="white")
    st.pyplot(fig)

# ==============================
# CONCLUSIONES
# ==============================
with tabs[6]:
    st.subheader("📝 Conclusiones")

    st.markdown("""
    - El PM2.5 presenta una tendencia estacional con mayores valores en invierno.  
    - Las variables meteorológicas (TEMP, PRES, DEWP) influyen en la dispersión del material particulado.  
    - La imputación redujo los valores faltantes sin alterar la estructura general de las distribuciones.  
    - Las correlaciones confirman la relación entre condiciones climáticas y contaminación atmosférica.  
    """)




