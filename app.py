import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Configuración de la página
st.set_page_config(page_title="Análisis EDA - Contaminación PRSA", layout="wide")

# Estilo general oscuro
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #FFFFFF;
}
h1, h2, h3, h4 {
    color: #00BFFF;
}
.analysis-box {
    background-color: rgba(0, 0, 0, 0.6);
    color: #FFFFFF;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228.csv")
    return df

df = load_data()
st.title("🌍 Análisis Exploratorio del Dataset PRSA - Contaminación y Clima")

# --- Tabs principales ---
tabs = st.tabs(["Resumen Ejecutivo", "Distribuciones", "Series Temporales",
                "Correlaciones", "Estacionalidad", "Datos Faltantes", "Análisis Bivariado", "Conclusiones"])

# --- Resumen Ejecutivo ---
with tabs[0]:
    st.header("📊 Resumen Ejecutivo")
    st.write(df.describe())
    st.markdown("""
    <div class="analysis-box">
    Este análisis explora los niveles de contaminación (PM2.5, PM10) y sus posibles relaciones con variables
    meteorológicas como temperatura, humedad, presión, punto de rocío y velocidad del viento.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Distribución General de PM2.5")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["PM2.5"], kde=True, color="skyblue", ax=ax)
    st.pyplot(fig)

# --- Distribuciones ---
with tabs[1]:
    st.header("📈 Distribuciones de Variables Meteorológicas")
    variables = ["TEMP", "PRES", "DEWP", "HUMI", "WSPM"]
    for var in variables:
        st.subheader(f"Distribución de {var}")
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(df[var], kde=True, color="orange", ax=ax)
        st.pyplot(fig)
    st.markdown("""
    <div class="analysis-box">
    Las variables meteorológicas muestran patrones distintos de dispersión y concentración.
    Por ejemplo, la temperatura y la humedad presentan una relación inversa a lo largo del año.
    </div>
    """, unsafe_allow_html=True)

# --- Series Temporales ---
with tabs[2]:
    st.header("⏱️ Series Temporales")
    if "year" not in df.columns:
        df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df_ts = df.set_index("date")

    st.subheader("Evolución de PM2.5 a lo largo del tiempo")
    fig, ax = plt.subplots(figsize=(10, 4))
    df_ts["PM2.5"].plot(ax=ax, color="cyan")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)

# --- Correlaciones ---
with tabs[3]:
    st.header("🔗 Matriz de Correlación")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    <div class="analysis-box">
    Se observa una alta correlación entre PM2.5 y PM10, lo que indica comportamientos similares en la concentración
    de contaminantes. Las variables meteorológicas muestran correlaciones moderadas con la contaminación, 
    sugiriendo influencia ambiental.
    </div>
    """, unsafe_allow_html=True)

# --- Estacionalidad ---
with tabs[4]:
    st.header("🌦️ Estacionalidad de los Contaminantes")
    df["month_name"] = df["month"].map({
        1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
        7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"
    })
    mean_month = df.groupby("month_name")[["PM2.5","PM10"]].mean()

    fig, ax = plt.subplots(figsize=(8,4))
    mean_month.plot(kind="bar", ax=ax, color=["cyan","orange"])
    ax.set_ylabel("Concentración promedio")
    st.pyplot(fig)

    st.markdown("""
    <div class="analysis-box">
    Se observa mayor concentración de contaminantes durante los meses fríos, lo que coincide con 
    un aumento en el uso de calefacción y menor dispersión atmosférica.
    </div>
    """, unsafe_allow_html=True)

# --- Datos Faltantes ---
with tabs[5]:
    st.header("🚧 Análisis de Datos Faltantes")
    st.subheader("Antes de imputar")
    missing = df.isna().mean() * 100
    st.bar_chart(missing)

    st.subheader("Tipo de ausencia")
    st.markdown("""
    <div class="analysis-box">
    En este contexto, la mayoría de las ausencias parecen ser de tipo **MAR (Missing At Random)**, 
    pues dependen de condiciones meteorológicas y fallas en los sensores.
    </div>
    """, unsafe_allow_html=True)

    # Imputación
    imp = SimpleImputer(strategy="mean")
    df_imputed = df.copy()
    df_imputed[["PM2.5", "PM10", "TEMP", "PRES", "DEWP", "HUMI", "WSPM"]] = imp.fit_transform(
        df_imputed[["PM2.5", "PM10", "TEMP", "PRES", "DEWP", "HUMI", "WSPM"]]
    )

    st.subheader("Después de imputar")
    missing_after = df_imputed.isna().mean() * 100
    st.bar_chart(missing_after)

# --- Análisis Bivariado ---
with tabs[6]:
    st.header("📉 Análisis Bivariado")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    x_var = st.selectbox("Selecciona variable X:", cols, index=0)
    y_var = st.selectbox("Selecciona variable Y:", cols, index=1)

    fig, ax = plt.subplots(figsize=(7,4))
    sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax, color="violet")
    st.pyplot(fig)

# --- Conclusiones ---
with tabs[7]:
    st.header("🧠 Conclusiones")
    st.markdown("""
    <div class="analysis-box">
    - La calidad del aire muestra patrones estacionales marcados.  
    - Las variables meteorológicas influyen directamente sobre la concentración de contaminantes.  
    - El manejo adecuado de datos faltantes es esencial para evitar sesgos.  
    - Este análisis puede ampliarse incorporando variables socioeconómicas o fuentes industriales.  
    </div>
    """, unsafe_allow_html=True)


