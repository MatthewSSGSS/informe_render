# app.py - Streamlit (corrección KeyError y EDA completo listo para Render)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

st.set_page_config(page_title="EDA PRSA - Dongsi", layout="wide")
st.markdown("""
    <style>
    body { background-color: #0f1720; color: #ffffff; }
    .block-container { padding: 1rem; }
    .analysis-box { background: rgba(255,255,255,0.03); padding:10px; border-radius:8px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load & normalize columns
# -------------------------
@st.cache_data
def load_data(path="PRSA_Data_Dongsi_20130301-20170228.csv"):
    df = pd.read_csv(path)
    # normalize column names: strip, lower, replace dots with underscore, replace spaces
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_', regex=False).str.replace(' ', '_', regex=False)
    # build datetime if year/month/day/hour present
    cols = df.columns
    if set(['year','month','day','hour']).issubset(cols):
        try:
            df['datetime'] = pd.to_datetime(dict(year=df['year'].astype(int),
                                                 month=df['month'].astype(int),
                                                 day=df['day'].astype(int),
                                                 hour=df['hour'].astype(int)))
        except Exception:
            # fallback: try combining as string
            df['datetime'] = pd.to_datetime(df[['year','month','day','hour']].astype(str).agg(' '.join, axis=1), errors='coerce')
    else:
        # if there's a date-like column try to parse it
        date_cols = [c for c in cols if 'date' in c]
        if date_cols:
            df['datetime'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("No se encontró 'PRSA_Data_Dongsi_20130301-20170228.csv' en la raíz. Súbelo y recarga la app.")
    st.stop()

# -------------------------
# Prepare variable lists
# -------------------------
all_cols = df.columns.tolist()
# candidate pollutant names and meteorological names (normalized)
candidates = {
    'pollutants': ['pm2_5','pm10','so2','no2','co','o3'],
    'mets': ['temp','pres','dewp','rain','wspm','humi','iws','is','ir','wd','cbwd']
}
pollutants = [c for c in candidates['pollutants'] if c in all_cols]
mets = [c for c in candidates['mets'] if c in all_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# keep original copy for before-imputation comparisons
df_original = df.copy()
df_imputed = None  # will be set after imputing

# -------------------------
# Sidebar controls - CORREGIDO
# -------------------------
st.sidebar.header("Controles")
tab_choice = st.sidebar.selectbox("Ir a sección", [
    "Resumen", "Distribuciones", "Series temporales",
    "Correlaciones", "Faltantes", "Bivariado", "Conclusiones"
])

# -------------------------
# Imputación automática - DECISIÓN TÉCNICA
# -------------------------
# Explicación de por qué elegimos interpolación temporal:
# 1. Los datos son de series de tiempo (fechas consecutivas)
# 2. La interpolación temporal preserva la estructura temporal
# 3. Es más apropiado que mean/median para datos secuenciales

def impute_dataframe(df_in):
    """Aplica imputación inteligente según el tipo de variable"""
    df_out = df_in.copy()
    
    if 'datetime' not in df_out.columns or df_out['datetime'].isna().all():
        st.warning("No hay columna datetime válida, usando métodos básicos")
        return impute_fallback(df_out)
    
    df_out = df_out.sort_values('datetime').set_index('datetime')
    
    # ESTRATEGIAS ESPECÍFICAS POR TIPO DE VARIABLE
    if 'pm2_5' in df_out.columns:
        df_out['pm2_5'] = df_out['pm2_5'].interpolate(method='time').ffill().bfill()
    
    if 'pm10' in df_out.columns:
        df_out['pm10'] = df_out['pm10'].interpolate(method='time').ffill().bfill()
    
    # Para gases (SO2, NO2, CO, O3) - interpolación temporal
    gases = ['so2', 'no2', 'co', 'o3']
    for gas in gases:
        if gas in df_out.columns:
            df_out[gas] = df_out[gas].interpolate(method='time').ffill().bfill()
    
    # Variables meteorológicas - estrategias específicas
    if 'temp' in df_out.columns:
        df_out['temp'] = df_out['temp'].interpolate(method='time')  # Temperatura tiene ciclos diarios
    
    if 'pres' in df_out.columns:
        df_out['pres'] = df_out['pres'].interpolate(method='time')  # Presión atmosférica
    
    if 'dewp' in df_out.columns:
        df_out['dewp'] = df_out['dewp'].interpolate(method='time')  # Punto de rocío
    
    if 'wspm' in df_out.columns:
        df_out['wspm'] = df_out['wspm'].fillna(0)  # Velocidad viento - 0 si no hay dato
    
    if 'wd' in df_out.columns:
        df_out['wd'] = df_out['wd'].ffill().bfill()  # Dirección viento - forward/backward fill
    
    if 'rain' in df_out.columns:
        df_out['rain'] = df_out['rain'].fillna(0)  # Lluvia - 0 si no hay dato (asumir no llueve)
    
    # Para cualquier otra variable numérica no cubierta
    remaining_numeric = df_out.select_dtypes(include=[np.number]).columns.difference(
        ['pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'temp', 'pres', 'dewp', 'wspm', 'rain']
    )
    for col in remaining_numeric:
        df_out[col] = df_out[col].interpolate(method='time').ffill().bfill()
    
    df_out = df_out.reset_index()
    return df_out

def impute_fallback(df_out):
    """Fallback cuando no hay datetime"""
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    
    # Estrategias básicas sin información temporal
    for col in numeric_cols:
        if col in ['rain', 'wspm']:
            df_out[col] = df_out[col].fillna(0)  # Lluvia y viento = 0 si falta
        elif col == 'wd':
            df_out[col] = df_out[col].ffill().bfill()  # Dirección viento
        else:
            # Para contaminantes y otras variables, usar mediana
            df_out[col] = df_out[col].fillna(df_out[col].median())
    
    return df_out

# Aplicar imputación automáticamente
df_imputed = impute_dataframe(df_original)

# -------------------------
# Heurística tipo de ausencia
# -------------------------
def classify_missing_type(col_name, df_obj):
    p = df_obj[col_name].isna().mean()
    if p == 0:
        return "Sin faltantes"
    
    # Obtener columnas numéricas excluyendo la columna actual si es numérica
    numeric = df_obj.select_dtypes(include=[np.number])
    
    # Si la columna actual es numérica, excluirla del análisis de correlación
    if col_name in numeric.columns:
        numeric = numeric.drop(columns=[col_name])
    
    # Si no hay columnas numéricas restantes, usar heurística simple
    if numeric.shape[1] == 0:
        return "MCAR" if p < 0.2 else "MNAR"
    
    # Calcular correlación entre indicador de missing y otras variables numéricas
    indicator = df_obj[col_name].isna().astype(int)
    cors = numeric.apply(lambda x: indicator.corr(x) if x.notna().any() else np.nan)
    max_abs_corr = cors.abs().max(skipna=True)
    
    if pd.isna(max_abs_corr):
        return "MCAR"
    if max_abs_corr > 0.3:
        return "MAR"
    if p > 0.20:
        return "MNAR"
    return "MCAR"

# -------------------------
# Sections (tabs)
# -------------------------
def show_summary():
    st.header("Resumen del dataset")
    st.markdown(f"- Filas: **{df_original.shape[0]}**  •  Columnas: **{df_original.shape[1]}**")
    st.markdown(f"- Variables numéricas detectadas: **{len(numeric_cols)}**")
    if 'station' in df_original.columns:
        st.markdown(f"- Estación: **{df_original['station'].unique().tolist()}**")
    st.write("Primeras filas:")
    st.dataframe(df_original.head())

    # monthly trend for pollutants if exist
    if 'datetime' in df_original.columns and len(pollutants) > 0:
        dfm = df_original.set_index('datetime').resample('M')[pollutants].mean().reset_index()
        fig = px.line(dfm, x='datetime', y=pollutants, title="Promedio mensual - contaminantes")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columna 'datetime' o no se detectaron contaminantes para serie mensual.")

def show_distributions():
    st.header("Distribuciones (univariadas)")
    vars_for_select = numeric_cols.copy()
    if not vars_for_select:
        st.warning("No se encontraron variables numéricas.")
        return
    sel = st.selectbox("Selecciona variable numérica", vars_for_select)
    col = sel
    data = df_imputed[col].dropna()
    fig = px.histogram(data, x=data, nbins=40, title=f"Histograma de {col}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Estadísticos básicos:")
    st.table(df_imputed[col].describe().to_frame().T)

def show_time_series():
    st.header("Series temporales")
    if 'datetime' not in df_imputed.columns:
        st.warning("No se detectó columna de fecha/hora (`datetime`).")
        return
    vars_ts = numeric_cols
    sel = st.selectbox("Selecciona variable para serie temporal", vars_ts, index=0)
    dfts = df_imputed.set_index('datetime').resample('D')[sel].mean().reset_index()
    fig = px.line(dfts, x='datetime', y=sel, title=f"Serie diaria de {sel}")
    st.plotly_chart(fig, use_container_width=True)

def show_correlations():
    st.header("Matriz de correlación (sólo columnas disponibles)")
    # choose reasonable subset for correlation (use intersection)
    candidate_corr = ['pm2_5','pm10','so2','no2','co','o3','temp','pres','dewp','wspm','iws','is','ir','humi','rain']
    cols_corr = [c for c in candidate_corr if c in df_imputed.columns]
    if not cols_corr:
        # fallback to numeric cols
        cols_corr = numeric_cols
    if not cols_corr:
        st.warning("No hay columnas para correlación.")
        return
    corr = df_imputed[cols_corr].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlación entre variables")
    st.plotly_chart(fig, use_container_width=True)

def show_missing():
    st.header("Análisis de valores faltantes")
    # Before imputation (original)
    st.subheader("Antes de imputar (original)")
    miss_before = df_original.isna().sum()
    miss_before = miss_before[miss_before > 0].sort_values(ascending=False)
    if not miss_before.empty:
        st.dataframe(miss_before.to_frame("n_missing"))
        # CORRECCIÓN: Crear el DataFrame correctamente
        miss_before_df = miss_before.reset_index()
        miss_before_df.columns = ['variable', 'missing']  # Renombrar directamente las columnas
        fig = px.bar(miss_before_df, x='variable', y='missing', title="Faltantes antes")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No hay faltantes en el dataset original.")

    # classify types
    st.subheader("Tipo de ausencia (heurística)")
    types = {col: classify_missing_type(col, df_original) for col in df_original.columns if df_original[col].isna().sum() > 0}
    if types:
        st.dataframe(pd.DataFrame(list(types.items()), columns=['variable','tipo']).set_index('variable'))
    else:
        st.info("No hay columnas con faltantes para clasificar.")

    # After imputation (current df_imputed)
    st.subheader("Después de imputar (actual)")
    miss_after = df_imputed.isna().sum()
    miss_after = miss_after[miss_after > 0].sort_values(ascending=False)
    if not miss_after.empty:
        st.dataframe(miss_after.to_frame("n_missing"))
        # CORRECCIÓN: Crear el DataFrame correctamente
        miss_after_df = miss_after.reset_index()
        miss_after_df.columns = ['variable', 'missing']  # Renombrar directamente las columnas
        fig2 = px.bar(miss_after_df, x='variable', y='missing', title="Faltantes después")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("No quedan faltantes tras la imputación seleccionada.")

    # KS tests for variables that had missing originally
    st.subheader("Prueba KS (antes vs después) — variables que tuvieron NA")
    had_na = [c for c in df_original.columns if df_original[c].isna().sum() > 0 and c in numeric_cols]
    if not had_na:
        st.info("No hay variables numéricas con NA en el original para comparar.")
    else:
        ks_rows = []
        for c in had_na:
            orig_vals = df_original[c].dropna().values
            new_vals = df_imputed[c].dropna().values
            if len(orig_vals) < 2 or len(new_vals) < 2:
                ks_rows.append((c, np.nan, np.nan, "insuficientes datos"))
                continue
            try:
                stat, p = ks_2samp(orig_vals, new_vals)
                note = "No cambio significativo" if p > 0.05 else "Cambio significativo"
                ks_rows.append((c, float(stat), float(p), note))
            except Exception as e:
                ks_rows.append((c, np.nan, np.nan, f"error: {e}"))
        ks_df = pd.DataFrame(ks_rows, columns=['variable','ks_stat','pvalue','nota']).set_index('variable')
        st.dataframe(ks_df)

def show_bivariate():
    st.header("Análisis bivariado")
    if len(numeric_cols) < 2:
        st.warning("Se requieren al menos dos variables numéricas.")
        return
    x = st.selectbox("Variable X", numeric_cols, index=0)
    y = st.selectbox("Variable Y", numeric_cols, index=1)
    plot_type = st.selectbox("Tipo de gráfico", ["scatter", "scatter + trendline", "hexbin (dense)"])
    if plot_type == "scatter":
        fig = px.scatter(df_imputed, x=x, y=y, opacity=0.6, title=f"{x} vs {y}")
    elif plot_type == "scatter + trendline":
        fig = px.scatter(df_imputed, x=x, y=y, trendline="lowess", opacity=0.5, title=f"{x} vs {y} (lowess)")
    else:
        # hexbin via plotly densitiy_heatmap
        fig = px.density_heatmap(df_imputed, x=x, y=y, nbinsx=40, nbinsy=40, title=f"Densidad {x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)
    # show correlation
    corr_val = df_imputed[x].corr(df_imputed[y])
    st.write(f"Coeficiente de correlación (Pearson) entre **{x}** y **{y}**: **{corr_val:.3f}**")

# -------------------------
# Dispatch according to selection
# -------------------------
if tab_choice == "Resumen":
    show_summary()
elif tab_choice == "Distribuciones":
    show_distributions()
elif tab_choice == "Series temporales":
    show_time_series()
elif tab_choice == "Correlaciones":
    show_correlations()
elif tab_choice == "Faltantes":
    show_missing()
elif tab_choice == "Bivariado":
    show_bivariate()
elif tab_choice == "Conclusiones":
    st.header("Conclusiones")
    st.markdown("""
### Estrategia de Imputación por Tipo de Variable

**Contaminantes (PM2.5, PM10, SO2, NO2, CO, O3):**
-  **Interpolación Temporal** - Preserva patrones estacionales y tendencias

**Variables Meteorológicas:**
-  **Temperatura, Presión, Punto de Rocío:** Interpolación Temporal (patrones cíclicos)
-  **Velocidad del Viento:** Relleno con 0 (asume calma cuando no hay dato)
-  **Lluvia:** Relleno con 0 (asume no llueve cuando no hay dato)
-  **Dirección del Viento:** Forward/Backward Fill (persistencia direccional)

**Justificación Científica:**
- Los contaminantes muestran alta autocorrelación temporal
- Las variables meteorológicas tienen comportamientos físicos específicos
- Evita introducir sesgos en análisis posteriores
""")
# Footer
st.markdown("---")





