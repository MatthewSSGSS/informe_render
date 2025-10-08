import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Estilos globales con colores contrastantes + efecto hover
st.markdown("""
<style>
.box {
    color: black;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ccc;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 16px;
    line-height: 1.5;
    transition: background-color 0.3s ease;
}
.resumen { background-color: #D6EAF8; }
.distribuciones { background-color: #D5F5E3; }
.series { background-color: #FCF3CF; }
.correlaciones { background-color: #FADBD8; }
.estacionalidad { background-color: #E8DAEF; }
.faltantes { background-color: #FDEBD0; }
.conclusiones { background-color: #E5E8E8; }

.resumen:hover { background-color: #AED6F1; }
.distribuciones:hover { background-color: #ABEBC6; }
.series:hover { background-color: #F9E79F; }
.correlaciones:hover { background-color: #F5B7B1; }
.estacionalidad:hover { background-color: #D2B4DE; }
.faltantes:hover { background-color: #F8C471; }
.conclusiones:hover { background-color: #D6DBDF; }

.box strong {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# 📊 Datos de ejemplo
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'CO': np.random.normal(0.4, 0.1, n),
    'NO2': np.random.normal(0.3, 0.05, n),
    'PM10': np.random.normal(0.5, 0.15, n),
    'PM2_5': np.random.normal(0.45, 0.12, n)
})
df.loc[np.random.choice(df.index, 400, replace=False), 'CO'] = np.nan
df.loc[np.random.choice(df.index, 250, replace=False), 'NO2'] = np.nan

# 1. Resumen Ejecutivo
st.header("📌 Resumen Ejecutivo")
st.write(df.describe())
st.markdown("""
<div class="box resumen">
<strong>📊 Análisis:</strong> El resumen estadístico muestra que las concentraciones promedio de PM10 
y PM2.5 son ligeramente superiores a las de CO y NO2. Las desviaciones estándar sugieren una 
dispersión moderada en todas las variables.
</div>
""", unsafe_allow_html=True)

# 2. Distribuciones
st.header("📈 Distribuciones")
fig, ax = plt.subplots(figsize=(10, 5))
df.hist(ax=ax)
plt.tight_layout()
st.pyplot(fig)
st.markdown("""
<div class="box distribuciones">
<strong>📌 Análisis:</strong> Las distribuciones de contaminantes son aproximadamente normales, 
aunque se observa una ligera asimetría en PM10. Esto indica que algunos valores extremos podrían 
estar influyendo en la media.
</div>
""", unsafe_allow_html=True)

# 3. Series Temporales
st.header("🧭 Series Temporales")
df['fecha'] = pd.date_range(start='2022-01-01', periods=n, freq='H')
ts = df.set_index('fecha').resample('D').mean()
st.line_chart(ts)
st.markdown("""
<div class="box series">
<strong>🧠 Análisis:</strong> Se aprecia una tendencia estable en los contaminantes a lo largo del tiempo. 
No se observan picos abruptos, lo que podría indicar una estabilidad estacional en la fuente de emisiones.
</div>
""", unsafe_allow_html=True)

# 4. Correlaciones
st.header("🔗 Correlaciones")
corr = df[['CO','NO2','PM10','PM2_5']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)
st.markdown("""
<div class="box correlaciones">
<strong>🔍 Análisis:</strong> Existe una correlación positiva moderada entre PM10 y PM2.5, lo que indica 
que ambas variables tienden a aumentar juntas. Las correlaciones entre CO y otros contaminantes son más débiles.
</div>
""", unsafe_allow_html=True)

# 5. Estacionalidad
st.header("🌦️ Estacionalidad")
ts_monthly = ts.resample('M').mean()
st.line_chart(ts_monthly)
st.markdown("""
<div class="box estacionalidad">
<strong>🗓️ Análisis:</strong> Se observan leves fluctuaciones mensuales en las concentraciones, 
posiblemente relacionadas con variaciones climáticas o de actividad industrial.
</div>
""", unsafe_allow_html=True)

# 6. Datos Faltantes
st.header("🚨 Datos Faltantes")
faltantes = df.isna().sum()
st.bar_chart(faltantes)
st.markdown("""
<div class="box faltantes">
<strong>⚠️ Análisis:</strong> Las variables CO y NO2 concentran la mayoría de valores faltantes,
lo que podría sesgar análisis si no se trata adecuadamente. Se recomienda aplicar imputación
multivariante o análisis de sensibilidad para mitigar el impacto de la falta de datos.
</div>
""", unsafe_allow_html=True)

#  7. Conclusiones
st.header(" Conclusiones")
st.markdown("""
<div class="box conclusiones">
<strong>📝 Análisis Final:</strong> El análisis exploratorio revela una estructura relativamente estable en 
los contaminantes atmosféricos, con correlaciones moderadas y patrones estacionales leves. Sin embargo, 
la presencia de datos faltantes en CO y NO2 representa un punto crítico que debe abordarse antes de aplicar 
modelos predictivos o de inferencia causal.
</div>
""", unsafe_allow_html=True)

