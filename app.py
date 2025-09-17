# -*- coding: utf-8 -*-
# App Streamlit — Análises por Tempo de Casa, Média/Dia e TMA

import io
import csv
import unicodedata
import warnings
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="TMA & Média por Tempo de Casa", layout="wide")

# ==========================
# Utilidades
# ==========================
def norm(s: str) -> str:
    s = str(s).strip()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return s.lower()

def minutes_to_hhmmss(m: Optional[float]) -> Optional[str]:
    if pd.isna(m):
        return None
    total_seconds = int(round(float(m) * 60))
    h = total_seconds // 3600
    rem = total_seconds % 3600
    mm = rem // 60
    ss = rem % 60
    return f"{h:02d}:{mm:02d}:{ss:02d}"

@st.cache_data(show_spinner=False)
def read_any(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    name = file_name.lower()
    if name.endswith(".csv"):
        sample = raw_bytes[:4096].decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except Exception:
            sep = ";"
        return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python")
    elif name.endswith((".xls", ".xlsx")):
        x = pd.ExcelFile(io.BytesIO(raw_bytes))
        for sh in x.sheet_names:
            df_try = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sh)
            if df_try.shape[0] > 0 and df_try.shape[1] > 0:
                st.info(f"✔️ Lendo a aba: {sh}")
                return df_try
        return pd.read_excel(io.BytesIO(raw_bytes))
    else:
        raise ValueError("Formato não suportado. Use .csv, .xls ou .xlsx.")

def pick_column(df: pd.DataFrame, alias_list) -> Optional[str]:
    low = {col.lower(): col for col in df.columns}
    normed = {norm(col): col for col in df.columns}
    for a in alias_list:
        if a in df.columns:
            return a
        if a.lower() in low:
            return low[a.lower()]
        if norm(a) in normed:
            return normed[norm(a)]
    for a in alias_list:
        for col in df.columns:
            if norm(a) in norm(col) or a.lower() in col.lower():
                return col
    return None

def trend_xy(agg_df: pd.DataFrame, cat_col: str, y_col: str, ordem: list) -> Tuple[np.ndarray, np.ndarray]:
    agg_df = agg_df.copy()
    cat_to_pos = {cat: i for i, cat in enumerate(ordem)}
    agg_df["x_pos"] = agg_df[cat_col].map(cat_to_pos)
    agg_df = agg_df.dropna(subset=["x_pos"])
    x = agg_df["x_pos"].values.astype(float)
    y = agg_df[y_col].values.astype(float)
    if len(agg_df) >= 2 and np.isfinite(y).sum() >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        lx = np.array([x.min(), x.max()], dtype=float)
        ly = slope * lx + intercept
        return lx, ly
    return np.array([]), np.array([])

def make_hover(base_df: pd.DataFrame, cols_to_try, extra_cols: Optional[Dict[str, object]] = None):
    hd = {}
    extra_cols = extra_cols or {}
    for c in cols_to_try:
        if c in base_df.columns:
            hd[c] = True
    for c, v in extra_cols.items():
        if c in base_df.columns:
            hd[c] = v if isinstance(v, (str, bool)) else True
    return hd

# ==========================
# Sidebar — Upload & Config
# ==========================
st.sidebar.title("⚙️ Configurações")
uploaded = st.sidebar.file_uploader("📂 Envie seu arquivo (.xlsx, .xls ou .csv)", type=["xlsx", "xls", "csv"])

st.sidebar.markdown("---")
st.sidebar.caption("Este app detecta automaticamente colunas comuns. Você pode ajustar manualmente abaixo.")

ordem_tempo_casa = [
    "0 a 3 meses",
    "3 meses a 1 ano",
    "1 a 2 anos",
    "2 a 3 anos",
    "3 a 4 anos",
    "4 a 5 anos",
    "acima de 5 anos",
]

map_tma_min = {
    "0 a 10 min": 5,
    "11 a 15 min": 13,
    "16 a 20 min": 18,
    "21 a 30 min": 25.5,
    "acima de 31 min": 35,
}

aliases = {
    "agente_email": ["agente email", "agente_email", "email do agente", "Agente Email", "Agente", "agente"],
    "lider": ["lider", "líder", "leader", "Líder", "Lider"],
    "media_por_dia": ["media por dia", "média por dia", "media/dia", "MÉDIA POR DIA", "MEDIA POR DIA", "media_por_dia"],
    "range_tma": ["range tma", "faixa tma", "tma faixa", "Range tma", "RANGE TMA", "range_tma"],
    "tempo_casa": ["range tempo de casa", "tempo de casa", "faixa tempo de casa", "Range tempo de casa", "Tempo de casa", "range_tempo_de_casa"],
}

# ==========================
# Corpo — Execução
# ==========================
st.title("📈 TMA & Média por Dia × Tempo de Casa")
st.write("Faça o upload de um arquivo com colunas de **tempo de casa**, **média por dia** e **range de TMA**. O app tentará detectar as colunas automaticamente.")

if uploaded is None:
    st.info("👆 Envie um arquivo para começar.")
    st.stop()

try:
    df = read_any(uploaded.name, uploaded.getvalue())
except Exception as e:
    st.error(f"Erro ao ler arquivo: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

col_agente = pick_column(df, aliases["agente_email"]) or "agente_email"
col_lider = pick_column(df, aliases["lider"]) or "Líder"
col_media = pick_column(df, aliases["media_por_dia"]) or "MÉDIA POR DIA"
col_tma = pick_column(df, aliases["range_tma"]) or "Range tma"
col_tcasa = pick_column(df, aliases["tempo_casa"]) or "Tempo de casa"

st.sidebar.subheader("🧭 Ajuste de Colunas Detectadas")
col_agente = st.sidebar.selectbox("Coluna: Agente Email", options=df.columns, index=df.columns.get_loc(col_agente) if col_agente in df.columns else 0)
col_lider = st.sidebar.selectbox("Coluna: Líder", options=df.columns, index=df.columns.get_loc(col_lider) if col_lider in df.columns else 0)
col_media = st.sidebar.selectbox("Coluna: Média por Dia", options=df.columns, index=df.columns.get_loc(col_media) if col_media in df.columns else 0)
col_tma = st.sidebar.selectbox("Coluna: Range TMA", options=df.columns, index=df.columns.get_loc(col_tma) if col_tma in df.columns else 0)
col_tcasa = st.sidebar.selectbox("Coluna: Tempo de Casa", options=df.columns, index=df.columns.get_loc(col_tcasa) if col_tcasa in df.columns else 0)

st.success(f"✔️ Colunas: Agente=**{col_agente}** | Líder=**{col_lider}** | Média/Dia=**{col_media}** | Range TMA=**{col_tma}** | Tempo de Casa=**{col_tcasa}**")

for c in [col_agente, col_lider, col_tma, col_tcasa]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

if col_tcasa in df.columns:
    df[col_tcasa] = pd.Categorical(df[col_tcasa], categories=ordem_tempo_casa, ordered=True)
else:
    st.error("Coluna de 'tempo de casa' não encontrada.")
    st.stop()

df[col_media] = pd.to_numeric(df[col_media], errors="coerce")
df["TMA_min"] = df[col_tma].map(map_tma_min)
df["TMA_hms"] = df["TMA_min"].apply(minutes_to_hhmmss)

agg_media = (
    df.groupby(col_tcasa, observed=True)
    .agg(media_atendimentos=(col_media, "mean"), qtd_agentes=(col_media, "count"))
    .reset_index()
    .dropna(subset=["media_atendimentos"])
)
agg_media["media_atendimentos_hms"] = agg_media["media_atendimentos"].apply(minutes_to_hhmmss)

agg_tma = (
    df.groupby(col_tcasa, observed=True)
    .agg(tma_medio_min=("TMA_min", "mean"), qtd_agentes=("TMA_min", "count"))
    .reset_index()
    .dropna(subset=["tma_medio_min"])
)
agg_tma["tma_medio_hms"] = agg_tma["tma_medio_min"].apply(minutes_to_hhmmss)

lx1, ly1 = trend_xy(agg_media, col_tcasa, "media_atendimentos", ordem_tempo_casa)
lx2, ly2 = trend_xy(agg_tma, col_tcasa, "tma_medio_min", ordem_tempo_casa)

# ==========================
# Gráficos
# ==========================
st.header("📊 Visões Agregadas por Tempo de Casa")

colA, colB = st.columns(2)

with colA:
    fig1 = px.scatter(
        agg_media,
        x=col_tcasa,
        y="media_atendimentos",
        size="qtd_agentes",
        hover_data={col_tcasa: True, "qtd_agentes": True, "media_atendimentos": ":.2f", "media_atendimentos_hms": True},
        title="Média de Atendimentos por Dia × Tempo de Casa (Agregado)",
        category_orders={col_tcasa: ordem_tempo_casa},
    )
    if lx1.size > 0:
        x_start = ordem_tempo_casa[int(round(lx1[0]))]
        x_end = ordem_tempo_casa[int(round(lx1[-1]))]
        fig1.add_trace(go.Scatter(x=[x_start, x_end], y=ly1, mode="lines", name="Linha de Tendência", line=dict(dash="dash")))
    fig1.update_layout(
        title_font_color="white",
        font_color="white",
        plot_bgcolor="#2F2F2F",
        paper_bgcolor="#2F2F2F",
        xaxis=dict(categoryorder="array", categoryarray=ordem_tempo_casa, gridcolor="gray", title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(gridcolor="gray", title="Média de atendimentos por dia", title_font=dict(color="white"), tickfont=dict(color="white")),
        height=520,
    )
    st.plotly_chart(fig1, use_container_width=True, theme=None)

with colB:
    fig2 = px.scatter(
        agg_tma,
        x=col_tcasa,
        y="tma_medio_min",
        size="qtd_agentes",
        hover_data={col_tcasa: True, "qtd_agentes": True, "tma_medio_hms": True},
        title="TMA (médio) × Tempo de Casa (Agregado)",
        category_orders={col_tcasa: ordem_tempo_casa},
    )
    if lx2.size > 0:
        x_start = ordem_tempo_casa[int(round(lx2[0]))]
        x_end = ordem_tempo_casa[int(round(lx2[-1]))]
        fig2.add_trace(go.Scatter(x=[x_start, x_end], y=ly2, mode="lines", name="Linha de Tendência", line=dict(dash="dash")))
    fig2.update_layout(
        title_font_color="white",
        font_color="white",
        plot_bgcolor="#2F2F2F",
        paper_bgcolor="#2F2F2F",
        xaxis=dict(categoryorder="array", categoryarray=ordem_tempo_casa, gridcolor="gray", title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(gridcolor="gray", title="TMA médio (minutos)", title_font=dict(color="white"), tickfont=dict(color="white")),
        height=520,
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None)

st.header("👤 Visões por Agente")

hover_agente_tma = make_hover(df, [col_agente, col_lider, col_media, col_tma, col_tcasa], extra_cols={"TMA_hms": True})
hover_agente_media = make_hover(df, [col_agente, col_lider, col_media, col_tma, col_tcasa])

colC, colD = st.columns(2)

with colC:
    fig3 = px.scatter(
        df,
        x=col_tcasa,
        y="TMA_min",
        hover_data=hover_agente_tma,
        title="Agentes × Range de TMA (X = Tempo de Casa, Y = TMA em minutos)",
        category_orders={col_tcasa: ordem_tempo_casa},
    )
    fig3.update_layout(
        title_font_color="white",
        font_color="white",
        plot_bgcolor="#2F2F2F",
        paper_bgcolor="#2F2F2F",
        xaxis=dict(categoryorder="array", categoryarray=ordem_tempo_casa, gridcolor="gray", title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(gridcolor="gray", title="TMA do agente (minutos)", title_font=dict(color="white"), tickfont=dict(color="white")),
        height=520,
    )
    st.plotly_chart(fig3, use_container_width=True, theme=None)

with colD:
    fig4 = px.scatter(
        df,
        x=col_tcasa,
        y=col_media,
        hover_data=hover_agente_media,
        title="Agentes × Média de Atendimentos (X = Tempo de Casa)",
        category_orders={col_tcasa: ordem_tempo_casa},
    )
    fig4.update_layout(
        title_font_color="white",
        font_color="white",
        plot_bgcolor="#2F2F2F",
        paper_bgcolor="#2F2F2F",
        xaxis=dict(categoryorder="array", categoryarray=ordem_tempo_casa, gridcolor="gray", title_font=dict(color="white"), tickfont=dict(color="white")),
        yaxis=dict(gridcolor="gray", title="Média de atendimentos por dia", title_font=dict(color="white"), tickfont=dict(color="white")),
        height=520,
    )
    st.plotly_chart(fig4, use_container_width=True, theme=None)

# Downloads
st.markdown("---")
st.subheader("⬇️ Exportações (CSV)")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Baixar agregação de Média/Dia",
        data=agg_media.to_csv(index=False).encode("utf-8"),
        file_name="agg_media_por_tempo_de_casa.csv",
        mime="text/csv",
    )
with col2:
    st.download_button(
        "Baixar agregação de TMA",
        data=agg_tma.to_csv(index=False).encode("utf-8"),
        file_name="agg_tma_por_tempo_de_casa.csv",
        mime="text/csv",
    )

st.caption("✅ Pronto. Carregue outro arquivo na barra lateral para atualizar as visões.")
