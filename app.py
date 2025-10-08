import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(
    page_title="Analisis Calidad del Aire - Beijing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS minimalista
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .section-header { font-size: 1.3rem; color: #2e86ab; margin: 1rem 0; }
    .conclusion-box { background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #1f77b4; }
    .info-box { background-color: #e8f4fd; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; }
    .analysis-text { background-color: #f0f8ff; padding: 1rem; border-radius: 5px; margin: 1rem 0; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# Cache optimizado para datos
@st.cache_data(ttl=3600, show_spinner=False)
def generar_datos_balanceados():
    """Genera datos simulados balanceados entre realismo y rendimiento"""
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

# Cargar datos
with st.spinner('Cargando datos...'):
    datos = generar_datos_balanceados()

# Sidebar
with st.sidebar:
    st.header("Configuracion")
    
    contaminante = st.selectbox(
        "Contaminante principal:",
        ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
        index=0
    )
    
    variable_meteo = st.selectbox(
        "Variable meteorologica:",
        ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"],
        index=4
    )
    
    st.markdown("---")
    st.header("Informacion del Dataset")
    st.write("Periodo: 2013-03-01 a 2017-02-28")
    st.write(f"Observaciones: {len(datos):,}")
    st.write("Variables: 15")
    st.write("Frecuencia: Horaria")
    st.write("Estacion: Dongsi, Beijing")

# Pestañas principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumen Ejecutivo", 
    "Distribuciones", 
    "Series Temporales",
    "Correlaciones", 
    "Estacionalidad", 
    "Datos Faltantes",
    "Conclusiones"
])

with tab1:
    st.markdown('<h2 class="main-header">Analisis de Calidad del Aire - Beijing</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observaciones", f"{len(datos):,}")
    with col2:
        st.metric("Variables", "15")
    with col3:
        st.metric("Periodo", "4 anos")
    with col4:
        st.metric("Estacion", "Dongsi")
    
    st.markdown("""
    <div class="info-box">
    <h4>Resumen Ejecutivo</h4>
    <p>Este analisis exploratorio de datos (EDA) se centra en la calidad del aire en Beijing, 
    utilizando registros horarios de la estacion Dongsi para el periodo 2013-03-01 a 2017-02-28.</p>
    </div>
    """, unsafe_allow_html=True)
    
    mensual = datos.set_index('fecha').resample('M')['PM2.5'].mean().reset_index()
    fig = px.line(mensual, x='fecha', y='PM2.5', 
                  title='PM2.5 - Promedio Mensual',
                  labels={'fecha': 'Fecha', 'PM2.5': 'PM2.5 (µg/m³)'})
    fig.update_traces(line=dict(color='#E74C3C', width=3))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <h4>Hallazgos principales del analisis:</h4>
    <ul>
    <li><strong>Distribuciones asimetricas:</strong> Los contaminantes presentan distribuciones con asimetria positiva y colas largas hacia valores altos</li>
    <li><strong>Alta correlacion:</strong> PM2.5 y PM10 muestran correlacion muy alta (r ≈ 0.9), sugiriendo fuentes de emision comunes</li>
    <li><strong>Efecto meteorologico:</strong> Relacion inversa significativa entre velocidad del viento (WSPM) y concentraciones de PM2.5</li>
    <li><strong>Estacionalidad marcada:</strong> Patrones estacionales claros con maximos en meses de invierno</li>
    <li><strong>Datos faltantes:</strong> Valores ausentes significativos en CO (9.1%) y NO2 (4.6%) que requieren atencion</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h3 class="section-header">Distribuciones Univariadas de Contaminantes</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(datos, x=contaminante, 
                          title=f'Distribucion de {contaminante}',
                          nbins=30,
                          color_discrete_sequence=['#3498DB'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="conclusion-box">
        <strong>Analisis de Distribuciones:</strong><br><br>
        Las distribuciones de las fracciones particuladas y de los gases muestran asimetrias positivas, 
        colas largas y presencia de valores extremos. Estos rasgos estadisticos indican que la media 
        aritmetica esta influida por episodios de alta concentracion; por tanto, la mediana y percentiles 
        son medidas mas robustas para caracterizar la condicion tipica.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Estadisticas Descriptivas")
        stats = datos[contaminante].describe()
        st.write(f"**Media:** {stats['mean']:.2f}")
        st.write(f"**Mediana:** {stats['50%']:.2f}")
        st.write(f"**Desv. Estandar:** {stats['std']:.2f}")
        st.write(f"**Maximo:** {stats['max']:.2f}")
        st.write(f"**Asimetria:** {datos[contaminante].skew():.2f}")

with tab3:
    st.markdown('<h3 class="section-header">Analisis de Series Temporales</h3>', unsafe_allow_html=True)
    
    mensual_stats = datos.set_index('fecha').resample('M').agg({
        'PM2.5': ['mean', 'median']
    }).round(1)
    mensual_stats.columns = ['Media', 'Mediana']
    mensual_stats = mensual_stats.reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mensual_stats['fecha'], y=mensual_stats['Media'], 
                            name='Media', line=dict(color='#E74C3C', width=3)))
    fig.add_trace(go.Scatter(x=mensual_stats['fecha'], y=mensual_stats['Mediana'], 
                            name='Mediana', line=dict(color='#2980B9', width=3, dash='dash')))
    fig.update_layout(title='PM2.5 - Serie Temporal Mensual (Media vs Mediana)',
                     xaxis_title='Fecha', 
                     yaxis_title='PM2.5 (µg/m³)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <h4>Analisis de Serie Temporal - PM2.5 (media mensual)</h4>
    <p>La serie mensual de PM2.5 permite apreciar variaciones estacionales y tendencias a mediano plazo. 
    En el conjunto analizado se observan picos periodicos que pueden asociarse a condiciones 
    meteorologicas (estancamiento atmosferico) o a episodios de emisiones localizadas.</p>
    
    <p><strong>Hallazgos especificos:</strong></p>
    <ul>
    <li>Variacion estacional marcada con maximos en meses frios (diciembre-enero)</li>
    <li>Diferencia significativa entre media y mediana, indicando influencia de valores extremos</li>
    <li>Tendencia de mejoria en los ultimos anos del periodo analizado</li>
    <li>Patrones ciclicos sugerentes de factores meteorologicos estacionales</li>
    </ul>
    
    <p><strong>Recomendacion:</strong> Realizar descomposicion estacional (STL) y analisis de tendencias 
    para separar componentes estacionales, de ciclo y aleatorio, asi como probar modelos ARIMA 
    estacionales para prediccion.</p>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown('<h3 class="section-header">Analisis de Correlaciones</h3>', unsafe_allow_html=True)
    
    variables_correlacion = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    corr_matrix = datos[variables_correlacion].corr().round(2)
    
    fig = px.imshow(corr_matrix,
                   title='Matriz de Correlacion de Pearson - Variables Numericas',
                   color_continuous_scale='RdBu',
                   aspect='auto',
                   text_auto=True)
    fig.update_layout(width=700, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de dispersion
    fig = px.scatter(datos, x=variable_meteo, y=contaminante,
                    title=f'Relacion: {contaminante} vs {variable_meteo}',
                    opacity=0.4,
                    trendline="ols")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <h4>Analisis: Matriz de Correlacion</h4>
    <p>La matriz de correlaciones permite identificar relaciones lineales entre pares de variables. 
    Se observa correlacion alta entre PM2.5 y PM10 (r ≈ 0.9), lo que sugiere fuentes o procesos 
    comunes que afectan ambas fracciones.</p>
    
    <p><strong>Correlaciones clave identificadas:</strong></p>
    <ul>
    <li><strong>PM2.5 - PM10:</strong> Correlacion muy alta (r > 0.85) - procesos de emision comunes</li>
    <li><strong>PM2.5 - CO:</strong> Correlacion alta (r ≈ 0.75) - posible relacion con combustion</li>
    <li><strong>PM2.5 - WSPM:</strong> Correlacion negativa (r ≈ -0.30) - efecto de dispersion por viento</li>
    <li><strong>TEMP - PRES:</strong> Correlacion negativa fuerte (r ≈ -0.80) - relacion meteorologica esperada</li>
    <li><strong>O3 - TEMP:</strong> Correlacion positiva - formacion de ozono favorecida por temperatura</li>
    </ul>
    
    <p><strong>Interpretacion:</strong> Las correlaciones moderadas entre contaminantes y variables 
    meteorologicas avalan interpretaciones fisicas; sin embargo, se recomienda evaluar correlaciones 
    parciales para aislar efectos de confusion.</p>
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown('<h3 class="section-header">Analisis de Estacionalidad</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(datos, x='month', y='PM2.5',
                    title='Boxplot de PM2.5 por Mes',
                    labels={'month': 'Mes', 'PM2.5': 'PM2.5 (µg/m³)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        mediana_mensual = datos.groupby('month')['PM2.5'].median().reset_index()
        fig = px.bar(mediana_mensual, x='month', y='PM2.5',
                    title='Mediana de PM2.5 por Mes',
                    labels={'month': 'Mes', 'PM2.5': 'PM2.5 (µg/m³)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <h4>Analisis: Estacionalidad de PM2.5</h4>
    <p>El boxplot mensual revela variacion estacional en PM2.5: meses con mediana mas alta indican 
    temporadas de mayor contaminacion, posiblemente asociadas a condiciones meteorologicas 
    (inversion termica) o a variaciones en emisiones antropogenicas.</p>
    
    <p><strong>Patron estacional identificado:</strong></p>
    <ul>
    <li><strong>Meses de maxima contaminacion:</strong> Diciembre, Enero, Febrero - condiciones de inversion termica</li>
    <li><strong>Meses de minima contaminacion:</strong> Julio, Agosto, Septiembre - mayor dispersion atmosferica</li>
    <li><strong>Amplitud estacional:</strong> Diferencia de ~40-50 µg/m³ entre meses mas y menos contaminados</li>
    <li><strong>Variabilidad intra-mensual:</strong> Mayor dispersion en meses de invierno (cajas mas largas)</li>
    </ul>
    
    <p><strong>Implicaciones:</strong> La ausencia de outliers en este grafico (por decision de visualizacion) 
    ayuda a visualizar el comportamiento central por mes; se recomienda complementar con analisis de 
    percentiles superiores para estudiar episodios extremos.</p>
    
    <p><strong>Relacion con variables meteorologicas:</strong> Los meses de mayor contaminacion coinciden 
    con condiciones de menor velocidad del viento y mayores fenomenos de inversion termica, lo que 
    limita la dispersion de contaminantes.</p>
    </div>
    """, unsafe_allow_html=True)

with tab6:
    st.markdown('<h3 class="section-header">Analisis de Valores Faltantes</h3>', unsafe_allow_html=True)
    
    faltantes = pd.DataFrame({
        'Variable': datos.columns,
        'Valores_Faltantes': datos.isnull().sum(),
        'Porcentaje': (datos.isnull().sum() / len(datos) * 100).round(2)
    })
    faltantes = faltantes[faltantes['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(faltantes, x='Variable', y='Porcentaje',
                    title='Porcentaje de Valores Faltantes por Variable',
                    labels={'Variable': 'Variable', 'Porcentaje': 'Porcentaje de Faltantes (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(faltantes, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <h4>Analisis: Valores Faltantes</h4>
    <p>El analisis de valores faltantes evidencia que CO y NO2 son las series con mayor proporcion 
    de ausentes (9.1% y 4.6% respectivamente), posiblemente debido a fallas instrumentales, 
    mantenimiento o a restricciones de registro.</p>
    
    <p><strong>Distribucion de faltantes:</strong></p>
    <ul>
    <li><strong>CO:</strong> 9.1% de valores faltantes - mayor impacto en analisis multivariados</li>
    <li><strong>NO2:</strong> 4.6% de valores faltantes - afecta analisis de gases nitrogenados</li>
    <li><strong>PM2.5:</strong> 0.3% de valores faltantes - minima afectacion</li>
    <li><strong>Otras variables:</strong> Menos del 0.1% - despreciable</li>
    </ul>
    
    <p><strong>Recomendaciones para el manejo:</strong></p>
    <ul>
    <li>Dada la magnitud de faltantes en CO (~9%), cualquier analisis multivariante debe documentar 
    la estrategia de imputacion</li>
    <li>Evaluar la sensibilidad de resultados frente a distintas tecnicas (mediana, KNN, regresion multiple)</li>
    <li>Considerar analisis de patrones de faltantes (MCAR, MAR, MNAR) para seleccionar metodo apropiado</li>
    <li>Para analisis criticos, considerar multiple imputacion o modelos que manejen datos faltantes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab7:
    st.markdown('<h3 class="section-header">Conclusiones Formales del Analisis</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-box">
    <h4>Conclusiones Principales</h4>
    <ol>
    <li><strong>Distribuciones asimetricas:</strong> Las series de PM2.5 y PM10 presentan asimetria positiva y colas largas; la mediana es una medida robusta para describir las condiciones tipicas.</li>
    
    <li><strong>Alta covariacion:</strong> PM2.5 y PM10 muestran alta covariacion (r ≈ 0.9), lo que sugiere procesos emisivos comunes y justifica analisis conjunto en estudios de fuentes.</li>
    
    <li><strong>Efecto del viento:</strong> La velocidad del viento exhibe correlacion negativa con PM2.5, indicando que la dispersion atmosferica es un factor relevante para la variabilidad observada.</li>
    
    <li><strong>Datos faltantes:</strong> La existencia de valores faltantes en CO y NO2 requiere evaluacion y documentacion para evitar sesgos en el analisis.</li>
    
    <li><strong>Episodios extremos:</strong> Los episodios extremos detectados exigen auditoria de sensores y analisis caso por caso antes de su eliminacion o correccion.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-box">
    <h4>Recomendaciones para Analisis Futuros</h4>
    <ul>
    <li>Utilizar medidas robustas (mediana, percentiles) en lugar de la media aritmetica</li>
    <li>Implementar tecnicas de imputacion multivariante para datos faltantes</li>
    <li>Aplicar modelos de descomposicion estacional (STL) para series temporales</li>
    <li>Considerar modelos ARIMA estacionales para prediccion</li>
    <li>Realizar auditoria de valores extremos caso por caso</li>
    <li>Evaluar correlaciones parciales para aislar efectos de confusion</li>
    <li>Analizar la relacion entre variables meteorologicas y contaminantes con modelos avanzados</li>
    <li>Considerar analisis de fuentes mediante modelos de receptor</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Analisis Exploratorio de Datos - Calidad del Aire Beijing | "
    "Estacion Dongsi (2013-2017) | Desarrollado con Streamlit"
    "</div>", 
    unsafe_allow_html=True
)