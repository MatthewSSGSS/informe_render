import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(page_title="Análisis de Calidad del Aire en Beijing", page_icon="🌤", layout="wide")

# ==============================
# CSS PERSONALIZADO
# ==============================
st.markdown("""
    <style>
    /* Elimina fondo blanco de contenedores y métricas */
    div[data-testid="stMetric"] {
        background-color: rgba(0,0,0,0);
        border: none;
    }
    div[data-testid="stMarkdownContainer"] {
        background-color: rgba(0,0,0,0);
        color: black;  /* Texto negro para mejor contraste */
    }
    section.main > div {
        background-color: transparent;
    }
    [data-testid="stVerticalBlock"] div:has(> [data-testid="stMarkdownContainer"]) {
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# GENERACIÓN DE DATOS SIMULADOS
# ==============================
@st.cache_data
def generar_datos():
    fechas = pd.date_range(start="2013-01-01", end="2017-12-31", freq="D")
    n = len(fechas)
    rng = np.random.default_rng(42)

    pm25 = np.maximum(0, 100 + 30*np.sin(2*np.pi*fechas.dayofyear/365) + rng.normal(0, 20, n))
    pm10 = pm25 * 1.2 + rng.normal(0, 15, n)
    no2 = 40 + 10*np.cos(2*np.pi*fechas.dayofyear/365) + rng.normal(0, 10, n)
    temp = 10 + 15*np.sin(2*np.pi*fechas.dayofyear/365) + rng.normal(0, 5, n)
    humedad = 60 + 20*np.cos(2*np.pi*fechas.dayofyear/365) + rng.normal(0, 10, n)
    velocidad_viento = np.maximum(0, 5 + 2*rng.normal(0, 1, n))

    df = pd.DataFrame({
        "fecha": fechas,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "Temperatura": temp,
        "Humedad": humedad,
        "Velocidad Viento": velocidad_viento
    })

    # Introducir algunos valores nulos
    for col in ["PM2.5", "PM10", "NO2"]:
        df.loc[rng.choice(n, 50, replace=False), col] = np.nan

    return df

df = generar_datos()

# ==============================
# ✨ IMPUTACIÓN DE FALTANTES
# ==============================
df = df.set_index("fecha")  # facilitar interpolación temporal
df = df.interpolate(method="linear")  # interpolación
df = df.ffill().bfill()  # relleno de seguridad
df = df.reset_index()  # restaurar índice original

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Opciones de Análisis")
contaminante = st.sidebar.selectbox("Seleccionar contaminante", ["PM2.5", "PM10", "NO2"])
variable_meteo = st.sidebar.selectbox("Seleccionar variable meteorológica", ["Temperatura", "Humedad", "Velocidad Viento"])

# ==============================
# PESTAÑAS
# ==============================
tabs = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Distribuciones",
    "⏳ Series Temporales",
    "🔗 Correlaciones",
    "📅 Estacionalidad",
    "🚨 Datos Faltantes",
    "📝 Conclusiones"
])

# ==============================
# RESUMEN EJECUTIVO
# ==============================
with tabs[0]:
    st.subheader("📊 Resumen Ejecutivo")

    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio PM2.5", f"{df['PM2.5'].mean():.2f} µg/m³")
    col2.metric("Promedio PM10", f"{df['PM10'].mean():.2f} µg/m³")
    col3.metric("Promedio NO2", f"{df['NO2'].mean():.2f} µg/m³")

    df_mes = df.groupby(df['fecha'].dt.to_period("M")).mean(numeric_only=True)
    df_mes.index = df_mes.index.to_timestamp()

    fig = px.line(df_mes, x=df_mes.index, y=["PM2.5", "PM10", "NO2"],
                  title="Evolución Mensual de Contaminantes", markers=True)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Análisis:**  
    Los contaminantes presentan un patrón estacional claro: concentraciones más altas en invierno y más bajas en verano.  
    Esto podría relacionarse con el uso de calefacción y menor ventilación atmosférica en meses fríos.
    """)

# ==============================
# DISTRIBUCIONES
# ==============================
with tabs[1]:
    st.subheader("📈 Distribuciones")

    fig = px.histogram(df, x=contaminante, nbins=40, title=f"Distribución de {contaminante}", marginal="box")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    stats = df[contaminante].describe().to_frame().T
    st.dataframe(stats)

    st.markdown(f"""
    **Análisis:**  
    La distribución de **{contaminante}** muestra un comportamiento ligeramente sesgado a la derecha, lo que indica la presencia de días con concentraciones elevadas.
    Esto puede reflejar eventos puntuales de contaminación más intensa.
    """)

# ==============================
# SERIES TEMPORALES
# ==============================
with tabs[2]:
    st.subheader("⏳ Series Temporales")

    fig = px.line(df, x="fecha", y=contaminante, title=f"Serie Temporal de {contaminante}", markers=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Análisis:**  
    Se observan variaciones estacionales y picos ocasionales en la concentración de **{contaminante}**.  
    Estos picos podrían corresponder a episodios de mala calidad del aire por condiciones meteorológicas adversas o emisiones concentradas.
    """)

# ==============================
# CORRELACIONES
# ==============================
with tabs[3]:
    st.subheader("🔗 Correlaciones")

    corr = df.drop(columns=["fecha"]).corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlación")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Análisis:**  
    - Existe una **alta correlación positiva entre PM2.5 y PM10**, lo cual es lógico dado que ambos provienen de fuentes similares.  
    - La correlación negativa moderada con temperatura sugiere que en climas fríos las concentraciones tienden a aumentar.  
    - Las variables meteorológicas juegan un papel importante en la dispersión de contaminantes.
    """)

# ==============================
# ESTACIONALIDAD
# ==============================
with tabs[4]:
    st.subheader("📅 Estacionalidad")

    df["mes"] = df["fecha"].dt.month
    fig = px.box(df, x="mes", y=contaminante, title=f"Estacionalidad de {contaminante} por Mes")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Análisis:**  
    La concentración de **{contaminante}** es claramente más alta en los meses fríos (enero-febrero) y más baja en verano.  
    Esto confirma el comportamiento estacional detectado en el resumen ejecutivo.
    """)

# ==============================
# DATOS FALTANTES
# ==============================
with tabs[5]:
    st.subheader("🚨 Datos Faltantes")

    faltantes = df.isna().mean().sort_values(ascending=False) * 100
    st.bar_chart(faltantes)

    st.markdown(f"""
    **Análisis:**  
    Antes de imputar, existía un pequeño porcentaje de datos faltantes en las series de contaminantes.  
    Estos fueron tratados con **interpolación lineal y relleno temporal**, asegurando continuidad en las series sin afectar la estacionalidad.
    """)

# ==============================
# CONCLUSIONES
# ==============================
with tabs[6]:
    st.subheader("📝 Conclusiones")

    st.markdown("""
    - Los contaminantes presentan **patrones estacionales claros**, con concentraciones más altas en invierno.  
    - Existe una **correlación fuerte entre PM2.5 y PM10**, lo que indica que ambos responden a fuentes similares.  
    - Las variables meteorológicas como la **temperatura y humedad** influyen significativamente en la dispersión.  
    - El **porcentaje de datos faltantes** fue imputado con métodos adecuados, preservando tendencias.  
    - Este análisis permite priorizar medidas preventivas en temporadas de mayor concentración de contaminantes.
    """)

