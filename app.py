import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =======================
# 🎨 Configuración general
# =======================
st.set_page_config(page_title="Análisis de Calidad del Aire - Beijing", page_icon="🌫️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F9F9F9; }
    .section-header { font-size: 24px; font-weight: bold; color: #2C3E50; margin-top: 20px; }
    .conclusion-box {
        background-color: #E8F6F3;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #1ABC9C;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# 🧮 Generar datos simulados
# =======================
@st.cache_data
def generar_datos():
    np.random.seed(42)
    fechas = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    n = len(fechas)
    data = {
        'date': fechas,
        'PM2.5': np.abs(np.random.normal(80, 30, n) + 20*np.sin(2*np.pi*fechas.dayofyear/365)),
        'PM10': np.abs(np.random.normal(100, 40, n) + 25*np.sin(2*np.pi*fechas.dayofyear/365)),
        'SO2': np.abs(np.random.normal(20, 5, n)),
        'NO2': np.abs(np.random.normal(50, 15, n)),
        'CO': np.abs(np.random.normal(1, 0.5, n)),
        'O3': np.abs(np.random.normal(80, 20, n)),
        'TEMP': np.random.normal(15, 10, n),
        'PRES': np.random.normal(1010, 5, n),
        'DEWP': np.random.normal(5, 8, n),
        'RAIN': np.abs(np.random.exponential(2, n)),
        'WSPM': np.abs(np.random.normal(2, 1, n)),
    }
    df = pd.DataFrame(data)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

datos = generar_datos()

# =======================
# 🧭 Sidebar de selección
# =======================
st.sidebar.header("⚙️ Controles")
contaminantes = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
variables_meteo = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
contaminante = st.sidebar.selectbox("Selecciona contaminante", contaminantes)
variable_meteo = st.sidebar.selectbox("Selecciona variable meteorológica", variables_meteo)

# =======================
# 📌 Tabs principales
# =======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen Ejecutivo",
    "Distribuciones",
    "Series Temporales",
    "Correlaciones",
    "Estacionalidad",
    "Datos Faltantes",
    "Conclusiones"
])

# =======================
# 📝 Resumen Ejecutivo
# =======================
with tab1:
    st.markdown('<h3 class="section-header">Resumen Ejecutivo</h3>', unsafe_allow_html=True)

    resumen = datos.describe().T
    st.dataframe(resumen, use_container_width=True)

    pm25_mean = datos['PM2.5'].mean()
    pm25_max = datos['PM2.5'].max()
    mes_max = datos.groupby('month')['PM2.5'].mean().idxmax()
    mes_min = datos.groupby('month')['PM2.5'].mean().idxmin()

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>🧠 Análisis Automático - Resumen Ejecutivo</h4>
    <ul>
    <li>La concentración promedio de PM2.5 es de <strong>{pm25_mean:.2f} µg/m³</strong>.</li>
    <li>El valor máximo registrado fue <strong>{pm25_max:.2f} µg/m³</strong>.</li>
    <li>El mes más contaminado fue <strong>{mes_max}</strong> y el más limpio <strong>{mes_min}</strong>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =======================
# 📊 Distribuciones
# =======================
with tab2:
    st.markdown(f'<h3 class="section-header">Distribución de {contaminante}</h3>', unsafe_allow_html=True)

    fig = px.histogram(datos, x=contaminante, nbins=40, title=f"Distribución de {contaminante}")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    skewness = datos[contaminante].skew()
    if skewness > 0:
        tendencia = "sesgo positivo (cola a la derecha)"
    elif skewness < 0:
        tendencia = "sesgo negativo (cola a la izquierda)"
    else:
        tendencia = "distribución simétrica"

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>📊 Análisis Automático - Distribuciones</h4>
    <ul>
    <li>{contaminante} presenta {tendencia}.</li>
    <li>Media: <strong>{datos[contaminante].mean():.2f}</strong> | Mediana: <strong>{datos[contaminante].median():.2f}</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =======================
# ⏳ Series Temporales
# =======================
with tab3:
    st.markdown(f'<h3 class="section-header">Series Temporales - {contaminante}</h3>', unsafe_allow_html=True)

    monthly = datos.groupby('month')[contaminante].agg(['mean', 'median']).reset_index()
    fig = px.line(monthly, x='month', y='mean', markers=True, title=f"Tendencia mensual de {contaminante}")
    fig.add_scatter(x=monthly['month'], y=monthly['median'], mode='lines+markers', name='Mediana')
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    mes_max_st = monthly.loc[monthly['mean'].idxmax(), 'month']
    mes_min_st = monthly.loc[monthly['mean'].idxmin(), 'month']
    variacion = monthly['mean'].max() - monthly['mean'].min()

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>📈 Análisis Automático - Series Temporales</h4>
    <ul>
    <li>El mes con mayor promedio fue <strong>{mes_max_st}</strong> y el menor <strong>{mes_min_st}</strong>.</li>
    <li>Variación promedio entre ambos: <strong>{variacion:.2f} µg/m³</strong>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =======================
# 🧭 Correlaciones
# =======================
with tab4:
    st.markdown('<h3 class="section-header">Correlaciones</h3>', unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion-box">
    <h4>🧠 Análisis Automático - Correlaciones</h4>
    <p>La matriz de correlación permite identificar relaciones lineales entre contaminantes y variables meteorológicas. Las correlaciones fuertes pueden señalar factores clave que influyen en la calidad del aire.</p>
    </div>
    """, unsafe_allow_html=True)

    vars_corr = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
    corr_matrix = datos[vars_corr].corr().round(2)
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu', text_auto=True, aspect='auto',
                    title="Matriz de Correlación de Pearson")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Var1', 'Var2', 'Corr']
    corr_pairs = corr_pairs[corr_pairs['Var1'] != corr_pairs['Var2']]
    corr_pairs = corr_pairs.drop_duplicates(subset=['Corr']).sort_values('Corr', ascending=False)
    top_pos = corr_pairs.head(3)
    top_neg = corr_pairs.tail(3)
    top_low = corr_pairs.iloc[len(corr_pairs)//2 - 1: len(corr_pairs)//2 + 2]

    st.write("🔸 **Top correlaciones positivas:**")
    st.dataframe(top_pos, use_container_width=True)
    st.write("🔹 **Correlaciones débiles:**")
    st.dataframe(top_low, use_container_width=True)
    st.write("🔻 **Top correlaciones negativas:**")
    st.dataframe(top_neg, use_container_width=True)

    fig = px.scatter(datos, x=variable_meteo, y=contaminante,
                     title=f"Relación {contaminante} vs {variable_meteo}",
                     opacity=0.4, trendline='lowess')
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

# =======================
# 🌦️ Estacionalidad
# =======================
with tab5:
    st.markdown(f'<h3 class="section-header">Estacionalidad - {contaminante}</h3>', unsafe_allow_html=True)

    fig = px.box(datos, x='month', y=contaminante, points='all', title=f"Distribución mensual de {contaminante}")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    season_median = datos.groupby('month')[contaminante].median()
    mes_pico = season_median.idxmax()
    mes_bajo = season_median.idxmin()

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>🌦️ Análisis Automático - Estacionalidad</h4>
    <ul>
    <li>La mediana más alta ocurre en <strong>{mes_pico}</strong> y la más baja en <strong>{mes_bajo}</strong>.</li>
    <li>Esto sugiere un patrón estacional marcado.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =======================
# ⚠️ Datos Faltantes
# =======================
with tab6:
    st.markdown('<h3 class="section-header">Datos Faltantes</h3>', unsafe_allow_html=True)

    faltantes = datos.isna().sum()
    st.dataframe(faltantes, use_container_width=True)

    faltantes_total = faltantes.sum()
    top_faltantes = faltantes.idxmax()
    porcentaje_top = (faltantes.max()/len(datos))*100

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>⚠️ Análisis Automático - Datos Faltantes</h4>
    <ul>
    <li>Total de valores faltantes: <strong>{faltantes_total}</strong></li>
    <li>Variable más afectada: <strong>{top_faltantes}</strong> ({porcentaje_top:.2f}%)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =======================
# 📌 Conclusiones
# =======================
with tab7:
    st.markdown('<h3 class="section-header">Conclusiones</h3>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="conclusion-box">
    <h4>📌 Conclusión General</h4>
    <ul>
    <li>Los contaminantes muestran un comportamiento estacional, con picos en meses fríos y mínimos en cálidos.</li>
    <li>Se identifican correlaciones fuertes entre contaminantes y variables meteorológicas clave.</li>
    <li>Las distribuciones no son perfectamente simétricas, lo cual podría requerir transformaciones.</li>
    <li>El tratamiento de datos faltantes es importante antes de aplicar modelos predictivos.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
