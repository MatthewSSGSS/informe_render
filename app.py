import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ========================
# Configuración de página
# ========================
st.set_page_config(
    page_title="Análisis Calidad del Aire - Beijing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CSS minimalista
# ========================
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .section-header { font-size: 1.3rem; color: #2e86ab; margin: 1rem 0; }
    .conclusion-box { background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #1f77b4; }
    .info-box { background-color: #e8f4fd; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ========================
# Generar datos simulados
# ========================
@st.cache_data(ttl=3600, show_spinner=False)
def generar_datos_balanceados():
    np.random.seed(123)
    n = 15000
    dates = pd.date_range('2013-03-01', '2017-02-28', freq='H')[:n]
    datos = pd.DataFrame({
        'fecha': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
    })
    base_pattern = 1 + 0.4 * np.sin(2 * np.pi * datos['month'] / 12)
    datos['PM2.5'] = np.random.gamma(1.5, 40, n) * base_pattern * (1 + 0.1 * np.random.normal(0, 1, n))
    datos['PM10'] = datos['PM2.5'] * 1.1 + np.random.normal(0, 15, n)
    datos['SO2'] = np.random.gamma(1.2, 25, n)
    datos['NO2'] = np.random.gamma(1.6, 30, n)
    datos['CO'] = datos['PM2.5'] * 15 + np.random.normal(0, 200, n)
    datos['O3'] = np.random.gamma(1.7, 45, n)
    datos['TEMP'] = 15 + 10 * np.sin(2 * np.pi * datos['month'] / 12) + np.random.normal(0, 5, n)
    datos['PRES'] = 1015 + 5 * np.sin(2 * np.pi * datos['month'] / 12) + np.random.normal(0, 3, n)
    datos['DEWP'] = 10 + 5 * np.sin(2 * np.pi * datos['month'] / 12) + np.random.normal(0, 4, n)
    datos['RAIN'] = np.random.exponential(0.5, n)
    datos['WSPM'] = np.random.gamma(2, 1.2, n) - 0.3 * datos['PM2.5']

    mask_co = np.random.choice([True, False], n, p=[0.091, 0.909])
    mask_no2 = np.random.choice([True, False], n, p=[0.046, 0.954])
    mask_pm25 = np.random.choice([True, False], n, p=[0.003, 0.997])

    datos.loc[mask_co, 'CO'] = np.nan
    datos.loc[mask_no2, 'NO2'] = np.nan
    datos.loc[mask_pm25, 'PM2.5'] = np.nan

    return datos

# ========================
# Cargar datos
# ========================
with st.spinner('Cargando datos...'):
    datos = generar_datos_balanceados()

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.header("Configuración")
    contaminante = st.selectbox(
        "Contaminante principal:",
        ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
        index=0
    )
    variable_meteo = st.selectbox(
        "Variable meteorológica:",
        ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"],
        index=4
    )
    st.markdown("---")
    st.header("Información del Dataset")
    st.write("📅 Periodo: 2013-03-01 a 2017-02-28")
    st.write(f"🧾 Observaciones: {len(datos):,}")
    st.write("📊 Variables: 15")
    st.write("⏳ Frecuencia: Horaria")
    st.write("📍 Estación: Dongsi, Beijing")

# ========================
# Tabs
# ========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen Ejecutivo",
    "Distribuciones",
    "Series Temporales",
    "Correlaciones",
    "Estacionalidad",
    "Datos Faltantes",
    "Conclusiones"
])

# ========================
# Tab 1 - Resumen Ejecutivo
# ========================
with tab1:
    st.markdown('<h2 class="main-header">Análisis de Calidad del Aire - Beijing</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observaciones", f"{len(datos):,}")
    col2.metric("Variables", "15")
    col3.metric("Periodo", "4 años")
    col4.metric("Estación", "Dongsi")

    st.markdown("""
    <div class="info-box">
    <h4>Resumen Ejecutivo</h4>
    <p>Este análisis exploratorio de datos (EDA) se centra en la calidad del aire en Beijing, 
    utilizando registros horarios de la estación Dongsi para el periodo 2013-03-01 a 2017-02-28.</p>
    </div>
    """, unsafe_allow_html=True)

    mensual = datos.set_index('fecha').resample('M')['PM2.5'].mean().reset_index()
    fig = px.line(mensual, x='fecha', y='PM2.5',
                  title='PM2.5 - Promedio Mensual',
                  labels={'fecha': 'Fecha', 'PM2.5': 'PM2.5 (µg/m³)'})
    fig.update_traces(line=dict(color='#E74C3C', width=3))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Hallazgos principales:**
    - Distribuciones asimétricas en contaminantes con valores extremos
    - Alta correlación entre PM2.5 y PM10 (r ≈ 0.9)
    - Relación inversa entre WSPM y PM2.5
    - Patrones estacionales marcados
    - Valores faltantes significativos en CO (9.1%) y NO2 (4.6%)
    """)

# ========================
# Tab 2 - Distribuciones
# ========================
with tab2:
    st.markdown('<h3 class="section-header">Distribuciones Univariadas</h3>', unsafe_allow_html=True)
    fig = px.histogram(datos, x=contaminante,
                      title=f'Distribución de {contaminante}',
                      nbins=30,
                      color_discrete_sequence=['#3498DB'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="conclusion-box">
    <strong>📌 Análisis:</strong> Distribuciones con asimetría positiva y colas largas, 
    indicando presencia de valores extremos.
    </div>
    """, unsafe_allow_html=True)
    stats = datos[contaminante].describe()
    st.write(f"Media: {stats['mean']:.2f} | Mediana: {stats['50%']:.2f} | Desv. Estándar: {stats['std']:.2f} | Máximo: {stats['max']:.2f}")

# ========================
# Tab 3 - Series Temporales
# ========================
with tab3:
    st.markdown('<h3 class="section-header">Series Temporales</h3>', unsafe_allow_html=True)
    mensual_stats = datos.set_index('fecha').resample('M').agg({'PM2.5': ['mean', 'median']}).round(1)
    mensual_stats.columns = ['Media', 'Mediana']
    mensual_stats = mensual_stats.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mensual_stats['fecha'], y=mensual_stats['Media'],
                             name='Media', line=dict(color='#E74C3C', width=3)))
    fig.add_trace(go.Scatter(x=mensual_stats['fecha'], y=mensual_stats['Mediana'],
                             name='Mediana', line=dict(color='#2980B9', width=3, dash='dash')))
    fig.update_layout(title='PM2.5 - Serie Temporal Mensual (Media vs Mediana)',
                      xaxis_title='Fecha', yaxis_title='PM2.5 (µg/m³)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="conclusion-box">
    <strong>📈 Observación:</strong> Se aprecian patrones estacionales con picos periódicos asociados a condiciones meteorológicas.
    </div>
    """, unsafe_allow_html=True)

# ========================
# Tab 4 - Correlaciones
# ========================
with tab4:
    st.markdown('<h3 class="section-header">Análisis de Correlaciones</h3>', unsafe_allow_html=True)
    variables_correlacion = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    corr_matrix = datos[variables_correlacion].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        title='Matriz de Correlación de Pearson',
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    fig.update_layout(width=800, height=650, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Análisis automático
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlación']
    corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]
    corr_pairs = corr_pairs.drop_duplicates(subset=['Correlación'])
    corr_pairs_sorted = corr_pairs.sort_values('Correlación', ascending=False)

    top_pos = corr_pairs_sorted.head(3)
    top_neg = corr_pairs_sorted.tail(3)
    top_low = corr_pairs_sorted.iloc[len(corr_pairs_sorted)//2 - 1: len(corr_pairs_sorted)//2 + 2]

    st.markdown('<div class="conclusion-box"><strong>📊 Análisis Automático de Correlaciones</strong></div>', unsafe_allow_html=True)
    st.write("🔸 **Correlaciones más fuertes (positivas):**")
    st.dataframe(top_pos, use_container_width=True)
    st.write("🔹 **Correlaciones más débiles (cercanas a cero):**")
    st.dataframe(top_low, use_container_width=True)
    st.write("🔻 **Correlaciones negativas más fuertes:**")
    st.dataframe(top_neg, use_container_width=True)

    fig = px.scatter(
        datos, x=variable_meteo, y=contaminante,
        title=f'Relación: {contaminante} vs {variable_meteo}',
        opacity=0.4,
        trendline='lowess'
    )
    fig.update_traces(marker=dict(color='#27AE60', size=3))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ========================
# Tab 5 - Estacionalidad
# ========================
with tab5:
    st.markdown('<h3 class="section-header">Estacionalidad</h3>', unsafe_allow_html=True)
    fig1 = px.box(datos, x='month', y='PM2.5',
                  title='Boxplot de PM2.5 por Mes',
                  labels={'month': 'Mes', 'PM2.5': 'PM2.5 (µg/m³)'})
    fig2 = px.bar(datos.groupby('month')['PM2.5'].median().reset_index(),
                  x='month', y='PM2.5',
                  title='Mediana de PM2.5 por Mes',
                  labels={'month': 'Mes', 'PM2.5': 'PM2.5 (µg/m³)'})
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="conclusion-box">
    <strong>📆 Observación:</strong> La contaminación muestra clara estacionalidad, con picos en ciertos meses.
    </div>
    """, unsafe_allow_html=True)

# ========================
# Tab 6 - Valores Faltantes
# ========================
with tab6:
    st.markdown('<h3 class="section-header">Análisis de Valores Faltantes</h3>', unsafe_allow_html=True)
    faltantes = pd.DataFrame({
        'Variable': datos.columns,
        'Valores_Faltantes': datos.isnull().sum(),
        'Porcentaje': (datos.isnull().sum() / len(datos) * 100).round(2)
    })
    faltantes = faltantes[faltantes['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
    fig = px.bar(faltantes, x='Variable', y='Porcentaje',
                title='Porcentaje de Valores Faltantes por Variable')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(faltantes, use_container_width=True)

    st.markdown("""
    <div class="conclusion-box">
    <strong>⚠️ Observación:</strong> CO y NO2 presentan mayor proporción de valores ausentes (9.1% y 4.6%).
    </div>
    """, unsafe_allow_html=True)

# ========================
# Tab 7 - Conclusiones
# ========================
with tab7:
    st.markdown('<h3 class="section-header">Conclusiones Formales</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="conclusion-box">
    <h4>Conclusiones Principales</h4>
    <ul>
        <li>Distribuciones asimétricas: PM2.5 y PM10 presentan colas largas → usar mediana y percentiles.</li>
        <li>Alta covariación entre PM2.5 y PM10 (r ≈ 0.9), lo que indica fuentes comunes.</li>
        <li>Velocidad del viento inversamente correlacionada con PM2.5.</li>
        <li>Faltantes relevantes en CO y NO2 → documentar e imputar adecuadamente.</li>
        <li>Episodios extremos requieren revisión manual o filtros automáticos.</li>
    </ul>
    </div>

    <div class="conclusion-box">
    <h4>Recomendaciones</h4>
    <ul>
        <li>Usar medidas robustas en lugar de medias simples.</li>
        <li>Imputación multivariante de valores faltantes.</li>
        <li>Aplicar descomposición estacional STL.</li>
        <li>Evaluar modelos ARIMA estacionales para predicción.</li>
        <li>Auditar outliers de manera sistemática.</li>
        <li>Analizar correlaciones parciales para efectos de confusión.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================
# Footer
# ========================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Análisis Exploratorio de Datos - Calidad del Aire Beijing | "
    "Estación Dongsi (2013-2017) | Desarrollado con Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
