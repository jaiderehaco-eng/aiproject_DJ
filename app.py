"""
================================================================================
  DASHBOARD BURSÁTIL — Corficolombiana (CORFICOLCF.CL)
  Bolsa de Valores de Colombia | Regresión Lineal Múltiple
  Datos obtenidos directamente desde Yahoo Finance via yfinance
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corficolombiana BVC — Análisis Bursátil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Fuentes */
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Fondo general */
  .stApp {
    background: #0D1117;
    color: #E6EDF3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
  }

  /* Tarjetas KPI */
  .kpi-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
  }
  .kpi-card:hover { transform: translateY(-2px); }
  .kpi-label {
    font-size: 12px;
    font-weight: 500;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    font-weight: 400;
    color: #58A6FF;
    margin-bottom: 4px;
  }
  .kpi-sub {
    font-size: 11px;
    color: #6E7681;
  }

  /* Título principal */
  .main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    color: #E6EDF3;
    margin: 0;
    line-height: 1.1;
  }
  .main-subtitle {
    font-size: 16px;
    color: #8B949E;
    margin-top: 8px;
  }

  /* Sección headers */
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #E6EDF3;
    border-left: 3px solid #58A6FF;
    padding-left: 12px;
    margin: 32px 0 16px;
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
  }
  .badge-blue { background: #1F4B8E; color: #79C0FF; }
  .badge-green { background: #1B3A2B; color: #56D364; }
  .badge-orange { background: #3B2800; color: #FFA657; }

  /* Conclusiones */
  .conclusion-box {
    background: linear-gradient(135deg, #0D2137 0%, #0D1D33 100%);
    border: 1px solid #1F4B8E;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
  }
  .conclusion-title {
    font-family: 'DM Serif Display', serif;
    font-size: 18px;
    color: #79C0FF;
    margin-bottom: 12px;
  }
  .conclusion-text { color: #C9D1D9; font-size: 14px; line-height: 1.7; }

  /* Fórmula */
  .formula-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'Courier New', monospace;
    font-size: 15px;
    color: #56D364;
    text-align: center;
    margin: 12px 0;
  }

  /* Métricas */
  .metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 12px 0;
  }
  .metric-item {
    flex: 1;
    min-width: 140px;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
  }
  .metric-name { font-size: 11px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; }
  .metric-val { font-size: 22px; color: #79C0FF; font-weight: 600; margin-top: 4px; }

  /* Tabla de datos */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Separador */
  .separator { border: none; border-top: 1px solid #21262D; margin: 32px 0; }

  /* Streamlit overrides */
  h1, h2, h3 { color: #E6EDF3 !important; }
  .stSelectbox label { color: #8B949E !important; }
  .stSlider label { color: #8B949E !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS (con caché para no repetir petición)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(ticker: str, periodo: str) -> pd.DataFrame:
    """Descarga datos desde Yahoo Finance y limpia el DataFrame."""
    raw = yf.download(ticker, period=periodo, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    periodo = st.selectbox(
        "📅 Período de análisis",
        options=["6mo", "1y", "2y"],
        index=1,
        format_func=lambda x: {"6mo": "6 meses (~126 días)", "1y": "1 año (~252 días)", "2y": "2 años (~504 días)"}[x],
    )

    test_size = st.slider("🔬 % datos de prueba", 10, 30, 20, step=5)

    st.markdown("---")
    st.markdown("**🏢 Acción analizada**")
    st.markdown("""
    <div style='font-size:13px; color:#8B949E;'>
    <b style='color:#79C0FF;'>Corficolombiana S.A.</b><br>
    Ticker: <code>CORFICOLCF.CL</code><br>
    Bolsa de Valores de Colombia<br>
    Moneda: COP (Peso Colombiano)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📚 Variables del modelo**")
    st.markdown("""
    <div style='font-size:13px; color:#8B949E;'>
    <span class='badge badge-blue'>Y</span> <b style='color:#C9D1D9'>Close</b> — Variable dependiente<br><br>
    <span class='badge badge-green'>X₁</span> <b style='color:#C9D1D9'>Open</b> — Variable regresora 1<br>
    <span class='badge badge-orange'>X₂</span> <b style='color:#C9D1D9'>Volume</b> — Variable regresora 2
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 Descargando datos desde Yahoo Finance..."):
    df = cargar_datos("CORFICOLCF.CL", periodo)

if df.empty or len(df) < 30:
    st.error("❌ No se pudieron obtener datos. Revisa la conexión o cambia el período.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MODELO DE REGRESIÓN (calculado antes de mostrar)
# ─────────────────────────────────────────────────────────────────────────────
X = df[["Open", "Volume"]]
y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, shuffle=False
)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred_test  = modelo.predict(X_test)
y_pred_train = modelo.predict(X_train)

r2_tr   = r2_score(y_train, y_pred_train)
r2_te   = r2_score(y_test, y_pred_test)
rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_te  = mean_absolute_error(y_test, y_pred_test)

r_open_close   = df["Open"].corr(df["Close"])
r_volume_close = df["Volume"].corr(df["Close"])

# Paleta de colores del dashboard
AZUL   = "#58A6FF"
VERDE  = "#56D364"
NARANJA = "#FFA657"
FONDO  = "#0D1117"
FONDO2 = "#161B22"
BORDE  = "#21262D"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=FONDO2,
    plot_bgcolor=FONDO,
    font=dict(family="DM Sans", color="#C9D1D9"),
    xaxis=dict(gridcolor=BORDE, showline=False, zeroline=False),
    yaxis=dict(gridcolor=BORDE, showline=False, zeroline=False),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ─────────────────────────────────────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.markdown("""
    <p class='main-title'>Corficolombiana<br><span style='color:#58A6FF;'>CORFICOLCF.CL</span></p>
    <p class='main-subtitle'>
      Análisis Bursátil · Regresión Lineal Múltiple · Bolsa de Valores de Colombia
      &nbsp;&nbsp;<span class='badge badge-blue'>Datos: Yahoo Finance</span>
      <span class='badge badge-green'>Moneda: COP</span>
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>📌 Indicadores Clave</p>", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

kpis = [
    (k1, "Registros", f"{len(df):,}", "días hábiles"),
    (k2, "Precio Promedio", f"${df['Close'].mean():,.0f}", "COP — Close"),
    (k3, "Precio Actual", f"${df['Close'].iloc[-1]:,.0f}", "COP — último dato"),
    (k4, "Volumen Promedio", f"{df['Volume'].mean()/1e6:.2f}M", "acciones/día"),
    (k5, "r(Open, Close)", f"{r_open_close:.4f}", "Pearson"),
    (k6, "r(Vol, Close)", f"{r_volume_close:.4f}", "Pearson"),
]
for col, label, value, sub in kpis:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{value}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — DATASET INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗂️ Ver Dataset Completo", expanded=False):
    st.markdown(f"**{len(df)} registros** | Período: `{df['Date'].min().date()}` → `{df['Date'].max().date()}`")
    df_show = df.copy()
    df_show["Date"] = df_show["Date"].dt.strftime("%Y-%m-%d")
    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
        df_show[c] = df_show[c].map("${:,.2f}".format)
    df_show["Volume"] = df_show["Volume"].map("{:,.0f}".format)
    st.dataframe(df_show, use_container_width=True, height=280)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — SERIE TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>📈 Evolución del Precio de Cierre</p>", unsafe_allow_html=True)

ma20 = df["Close"].rolling(20).mean()
ma50 = df["Close"].rolling(50).mean()

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df["Date"], y=df["Close"],
    mode="lines", name="Close",
    line=dict(color=AZUL, width=1.8),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
))
fig_ts.add_trace(go.Scatter(
    x=df["Date"], y=ma20,
    mode="lines", name="MA 20d",
    line=dict(color=NARANJA, width=1.5, dash="dot")
))
fig_ts.add_trace(go.Scatter(
    x=df["Date"], y=ma50,
    mode="lines", name="MA 50d",
    line=dict(color=VERDE, width=1.5, dash="dash")
))
# Máximo y mínimo
idx_max = df["Close"].idxmax()
idx_min = df["Close"].idxmin()
fig_ts.add_trace(go.Scatter(
    x=[df.loc[idx_max, "Date"]], y=[df["Close"].max()],
    mode="markers+text", name=f"Máx ${df['Close'].max():,.0f}",
    marker=dict(color=VERDE, size=10, symbol="triangle-up"),
    text=[f"  ${df['Close'].max():,.0f}"], textposition="top right",
    textfont=dict(color=VERDE, size=11)
))
fig_ts.add_trace(go.Scatter(
    x=[df.loc[idx_min, "Date"]], y=[df["Close"].min()],
    mode="markers+text", name=f"Mín ${df['Close'].min():,.0f}",
    marker=dict(color="#F85149", size=10, symbol="triangle-down"),
    text=[f"  ${df['Close'].min():,.0f}"], textposition="bottom right",
    textfont=dict(color="#F85149", size=11)
))
fig_ts.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(text="Corficolombiana — Serie Temporal del Precio de Cierre (COP)", font=dict(size=15)),
    yaxis_tickprefix="$",
    yaxis_tickformat=",.0f",
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDE),
    height=380,
)
st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — SCATTER PLOTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>🔵 Relación con Variables Regresoras</p>", unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

# — Scatter Open vs Close —
with col_s1:
    fig_oc = px.scatter(
        df, x="Open", y="Close",
        color=df.index,
        color_continuous_scale="Blues",
        trendline="ols",
        labels={"Open": "Precio Apertura (COP)", "Close": "Precio Cierre (COP)"},
        title=f"<b>Open vs Close</b> &nbsp;|&nbsp; r = {r_open_close:.4f}"
    )
    fig_oc.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0.3, color="#161B22")))
    fig_oc.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        coloraxis_showscale=False,
        xaxis_tickprefix="$", xaxis_tickformat=",.0f",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_oc, use_container_width=True)
    st.markdown(f"""
    <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;padding:12px 16px;font-size:13px;color:#C9D1D9;'>
      <b style='color:{AZUL}'>r = {r_open_close:.4f}</b> &nbsp;|&nbsp;
      <b style='color:{VERDE}'>R² = {r_open_close**2:.4f}</b> &nbsp;|&nbsp;
      Open explica el <b>{r_open_close**2*100:.1f}%</b> de la varianza de Close
    </div>
    """, unsafe_allow_html=True)

# — Scatter Volume vs Close —
with col_s2:
    df_plot = df.copy()
    df_plot["Volume_M"] = df_plot["Volume"] / 1e6
    fig_vc = px.scatter(
        df_plot, x="Volume_M", y="Close",
        color=df_plot.index,
        color_continuous_scale="Purples",
        trendline="ols",
        labels={"Volume_M": "Volumen (millones de acciones)", "Close": "Precio Cierre (COP)"},
        title=f"<b>Volume vs Close</b> &nbsp;|&nbsp; r = {r_volume_close:.4f}"
    )
    fig_vc.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0.3, color="#161B22")))
    fig_vc.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        coloraxis_showscale=False,
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_vc, use_container_width=True)
    st.markdown(f"""
    <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;padding:12px 16px;font-size:13px;color:#C9D1D9;'>
      <b style='color:{AZUL}'>r = {r_volume_close:.4f}</b> &nbsp;|&nbsp;
      <b style='color:{VERDE}'>R² = {r_volume_close**2:.4f}</b> &nbsp;|&nbsp;
      Volume explica el <b>{r_volume_close**2*100:.1f}%</b> de la varianza de Close
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — HEATMAP DE CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>🔥 Matriz de Correlaciones</p>", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([1, 1])

with col_h1:
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr().round(3)
    fig_hm = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        hoverongaps=False,
        showscale=True,
    ))
    fig_hm.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(text="Correlaciones de Pearson", font=dict(size=15)),
    height=360
)

    st.plotly_chart(fig_hm, use_container_width=True)

with col_h2:
    st.markdown("#### 📌 Interpretación")
    st.markdown(f"""
    <div style='color:#C9D1D9; font-size:14px; line-height:1.8;'>
    <b style='color:{VERDE}'>Verde oscuro → r ≈ +1</b>: Correlación positiva muy fuerte.<br>
    <b style='color:#F85149'>Rojo oscuro → r ≈ -1</b>: Correlación negativa fuerte.<br>
    <b style='color:{NARANJA}'>Amarillo → r ≈ 0</b>: Sin correlación lineal.<br><br>

    <b style='color:{AZUL}'>Hallazgos principales:</b><br>
    &bull; Open, High, Low y Close tienen correlaciones <b>≥ 0.98</b>, lo cual es
    coherente: todos son precios del mismo instrumento en el mismo día.<br><br>
    &bull; Volume muestra una correlación más baja con los precios, lo que
    refleja que el volumen no necesariamente sigue la dirección del precio
    en el mercado colombiano de renta variable.<br><br>
    &bull; La <b>multicolinealidad</b> entre Open y Close justifica el alto R²
    del modelo, pero también indica que Open es un proxy casi perfecto.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — REGRESIÓN LINEAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>🤖 Modelo de Regresión Lineal Múltiple</p>", unsafe_allow_html=True)

# Fórmula
signo = "+" if modelo.coef_[1] >= 0 else "-"
st.markdown(f"""
<div class='formula-box'>
  Close &nbsp;=&nbsp; {modelo.intercept_:.2f}
  &nbsp;+&nbsp; {modelo.coef_[0]:.6f} × Open
  &nbsp;{signo}&nbsp; {abs(modelo.coef_[1]):.10f} × Volume
</div>
""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("#### 📊 Métricas de Desempeño")
    metricas = {
        "R² Entrenamiento": f"{r2_tr:.4f}",
        "R² Prueba": f"{r2_te:.4f}",
        "RMSE Prueba (COP)": f"${rmse_te:,.2f}",
        "MAE Prueba (COP)": f"${mae_te:,.2f}",
        "Observaciones Train": f"{len(X_train):,}",
        "Observaciones Test": f"{len(X_test):,}",
    }
    for nombre, val in metricas.items():
        color = VERDE if "R²" in nombre else AZUL
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:8px 12px;
            background:#161B22;border:1px solid #21262D;border-radius:8px;margin:4px 0;'>
          <span style='color:#8B949E;font-size:13px;'>{nombre}</span>
          <span style='color:{color};font-weight:600;font-size:14px;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔢 Coeficientes")
    coefs = {
        "β₀ Intercepto": f"{modelo.intercept_:.4f} COP",
        "β₁ Open": f"{modelo.coef_[0]:.6f}",
        "β₂ Volume": f"{modelo.coef_[1]:.10f}",
    }
    for nombre, val in coefs.items():
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:8px 12px;
            background:#161B22;border:1px solid #21262D;border-radius:8px;margin:4px 0;'>
          <span style='color:#8B949E;font-size:13px;'>{nombre}</span>
          <span style='color:{NARANJA};font-weight:600;font-size:14px;font-family:monospace;'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

with col_m2:
    # Gráfico Real vs Predicho
    fechas_test = df["Date"].iloc[-len(y_test):].values
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(
        x=fechas_test, y=y_test.values,
        name="Close Real", mode="lines",
        line=dict(color=AZUL, width=2)
    ))
    fig_rp.add_trace(go.Scatter(
        x=fechas_test, y=y_pred_test,
        name="Close Predicho", mode="lines",
        line=dict(color=NARANJA, width=1.5, dash="dot")
    ))
    fig_rp.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Real vs Predicho | R² = {r2_te:.4f}", font=dict(size=14)),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=370,
    )
    st.plotly_chart(fig_rp, use_container_width=True)

st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — ESTADÍSTICAS DESCRIPTIVAS
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📐 Estadísticas Descriptivas", expanded=False):
    desc = df[["Open", "Close", "Volume"]].describe().round(2)
    desc_fmt = desc.copy().astype(object)
    for c in ["Open", "Close"]:
        for idx in desc_fmt.index:
            desc_fmt.loc[idx, c] = f"${desc.loc[idx, c]:,.2f}"
    for idx in desc_fmt.index:
        desc_fmt.loc[idx, "Volume"] = f"{float(desc.loc[idx, 'Volume']):,.0f}"
    st.dataframe(desc_fmt, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — CONCLUSIONES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>💡 Conclusiones del Análisis</p>", unsafe_allow_html=True)

# Determinar calidad del R²
if r2_te >= 0.90:
    r2_desc = "muy alto poder explicativo"
    r2_color = VERDE
elif r2_te >= 0.75:
    r2_desc = "buen poder explicativo"
    r2_color = AZUL
elif r2_te >= 0.50:
    r2_desc = "poder explicativo moderado"
    r2_color = NARANJA
else:
    r2_desc = "poder explicativo limitado"
    r2_color = "#F85149"

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown(f"""
    <div class='conclusion-box'>
      <div class='conclusion-title'>📌 Variable Regresora 1: Open</div>
      <div class='conclusion-text'>
        El precio de apertura presenta una correlación de <b style='color:{AZUL};'>
        r = {r_open_close:.4f}</b> con el precio de cierre, explicando el
        <b style='color:{VERDE};'>{r_open_close**2*100:.1f}%</b> de su variabilidad (R² individual).<br><br>
        Esta relación es esperada: en mercados con baja volatilidad intradía
        como la <b>Bolsa de Valores de Colombia</b>, el precio de apertura
        es un indicador robusto del precio de cierre, ya que las variaciones
        diarias suelen ser pequeñas respecto al precio base.
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown(f"""
    <div class='conclusion-box'>
      <div class='conclusion-title'>📌 Variable Regresora 2: Volume</div>
      <div class='conclusion-text'>
        El volumen de transacciones tiene una correlación de <b style='color:{AZUL};'>
        r = {r_volume_close:.4f}</b> con Close, aportando información sobre
        la <b>actividad del mercado</b>.<br><br>
        Aunque su poder explicativo individual es menor
        ({r_volume_close**2*100:.1f}%), el volumen complementa el modelo
        al capturar días de mayor actividad bursátil, los cuales suelen
        estar asociados a eventos corporativos o noticias relevantes de
        Corficolombiana en el mercado financiero colombiano.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class='conclusion-box' style='margin-top:16px; border-color:#30363D;'>
  <div class='conclusion-title' style='color:{r2_color};'>🏁 Conclusión General del Modelo</div>
  <div class='conclusion-text'>
    El modelo de <b>Regresión Lineal Múltiple</b>
    <code style='background:#0D2137;padding:2px 6px;border-radius:4px;'>
      Close = β₀ + β₁·Open + β₂·Volume
    </code>
    alcanza un <b style='color:{r2_color};'>R² = {r2_te:.4f}</b> en los datos de prueba,
    demostrando {r2_desc} para explicar el comportamiento del precio de cierre
    de Corficolombiana en la BVC.<br><br>
    La ecuación estimada es:
    <code style='background:#0D2137;padding:2px 6px;border-radius:4px;'>
      Close ≈ {modelo.intercept_:.2f} + {modelo.coef_[0]:.4f}·Open + {modelo.coef_[1]:.8f}·Volume
    </code><br><br>
    Período analizado: <b>{df['Date'].min().date()}</b> → <b>{df['Date'].max().date()}</b>
    &nbsp;·&nbsp; <b>{len(df)} días hábiles</b>
    &nbsp;·&nbsp; Datos obtenidos de <b>Yahoo Finance</b> en tiempo real.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='separator'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#484F58; font-size:12px; padding:8px 0 24px;'>
  Dashboard Bursátil — Corficolombiana (CORFICOLCF.CL) · Bolsa de Valores de Colombia<br>
  Datos: Yahoo Finance · Modelo: Regresión Lineal Múltiple · Desarrollado con Streamlit &amp; Python
</div>
""", unsafe_allow_html=True)
