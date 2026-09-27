"""
Entrega 4 — Buscador Semântico + App Streamlit
Desafio 1 · Ouvidoria Inteligente: Triagem Semântica de Manifestações Cidadãs

Executar com:
    pip install streamlit sentence-transformers scikit-learn langchain-text-splitters \
                pandas numpy matplotlib seaborn
    streamlit run app_ouvidoria.py

Espera encontrar "manifestacoes.json" na mesma pasta (ou envie pela barra lateral).
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

CAMINHO_JSON_PADRAO = "manifestacoes.json"

# Mesmos modelos usados nas Entregas 1-3 (BAAI/bge-small-pt-v1.5 não existe publicamente no HF)
MODELOS_DISPONIVEIS = {
    "MiniLM-L12 (paraphrase-multilingual-MiniLM-L12-v2)": "paraphrase-multilingual-MiniLM-L12-v2",
    "DistilUSE (distiluse-base-multilingual-cased-v2)": "distiluse-base-multilingual-cased-v2",
}

CORES_SCORE = [(0.7, "🟢"), (0.5, "🟡"), (0.0, "🔴")]


def emoji_score(score: float) -> str:
    for limite, emoji in CORES_SCORE:
        if score > limite:
            return emoji
    return "🔴"


# ----------------------------------------------------------------------------
# Cache: modelo (recurso) e embeddings (dados)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embedding...")
def carregar_modelo(nome_modelo: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nome_modelo)


@st.cache_data(show_spinner="Calculando embeddings da base...")
def calcular_embeddings(_modelo, textos: tuple, nome_modelo: str):
    # nome_modelo entra só para diferenciar a chave do cache entre modelos distintos
    return _modelo.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)


@st.cache_data
def carregar_dados(caminho_ou_bytes) -> pd.DataFrame:
    if hasattr(caminho_ou_bytes, "read"):
        dados = json.load(caminho_ou_bytes)
    else:
        with open(caminho_ou_bytes, encoding="utf-8") as f:
            dados = json.load(f)
    if isinstance(dados, dict):
        dados = next(v for v in dados.values() if isinstance(v, list))
    df = pd.DataFrame(dados)
    df["n_caracteres"] = df["texto"].str.len()
    return df


@st.cache_data
def reduzir_dimensionalidade(embeddings: np.ndarray, metodo: str) -> np.ndarray:
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    perplexidade = max(2, min(30, len(embeddings) - 1))
    return TSNE(n_components=2, random_state=42, perplexity=perplexidade, init="pca").fit_transform(embeddings)


def dividir_em_chunks(texto: str, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(texto)


# ----------------------------------------------------------------------------
# Barra lateral: dados, modelo, top-k
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações")

arquivo_enviado = st.sidebar.file_uploader("Base de manifestações (.json)", type=["json"])
try:
    df = carregar_dados(arquivo_enviado if arquivo_enviado is not None else CAMINHO_JSON_PADRAO)
except FileNotFoundError:
    st.sidebar.error(f"Não encontrei '{CAMINHO_JSON_PADRAO}'. Envie o arquivo JSON acima.")
    st.stop()

nome_modelo_exibicao = st.sidebar.selectbox("Modelo de embedding", list(MODELOS_DISPONIVEIS.keys()))
nome_modelo = MODELOS_DISPONIVEIS[nome_modelo_exibicao]
top_k = st.sidebar.slider("Top-k resultados na busca", min_value=1, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df)} manifestações carregadas · categorias: "
                    f"{', '.join(sorted(df['categoria_oficial'].unique()))}")

modelo = carregar_modelo(nome_modelo)
embeddings = calcular_embeddings(modelo, tuple(df["texto"].tolist()), nome_modelo)

st.title("🏛️ Ouvidoria Inteligente — Triagem Semântica de Manifestações")

aba_busca, aba_base, aba_espaco, aba_chunking = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)

# ----------------------------------------------------------------------------
# 🔍 Busca Semântica
# ----------------------------------------------------------------------------
with aba_busca:
    st.subheader("Busca por descrição livre")
    st.caption("O cidadão descreve o problema com suas próprias palavras; o sistema retorna as "
               "manifestações mais parecidas, mesmo sem compartilhar as mesmas palavras-chave.")

    consulta = st.text_area(
        "Descreva a manifestação",
        placeholder="Ex.: rua sem iluminação e com muitos assaltos à noite",
        height=90,
    )
    buscar = st.button("Buscar manifestações semelhantes", type="primary")

    if buscar and consulta.strip():
        vetor_consulta = modelo.encode([consulta], normalize_embeddings=True)
        scores = cosine_similarity(vetor_consulta, embeddings)[0]
        resultado = df.copy()
        resultado["score"] = scores
        resultado = resultado.sort_values("score", ascending=False).head(top_k)

        for _, linha in resultado.iterrows():
            emoji = emoji_score(linha["score"])
            with st.container(border=True):
                col_a, col_b = st.columns([5, 1])
                col_a.markdown(f"**{linha['id']}** · categoria oficial: `{linha['categoria_oficial']}`")
                col_a.write(linha["texto"])
                col_b.markdown(f"### {emoji} {linha['score']:.3f}")
        st.caption("🟢 similaridade > 0,70 · 🟡 > 0,50 · 🔴 demais")
    elif buscar:
        st.warning("Digite uma descrição antes de buscar.")

# ----------------------------------------------------------------------------
# 📋 Base Completa
# ----------------------------------------------------------------------------
with aba_base:
    st.subheader("Todas as manifestações")
    filtro_categoria = st.multiselect(
        "Filtrar por categoria oficial", sorted(df["categoria_oficial"].unique())
    )
    df_filtrado = df[df["categoria_oficial"].isin(filtro_categoria)] if filtro_categoria else df
    st.dataframe(
        df_filtrado[["id", "data", "categoria_oficial", "n_caracteres", "texto"]],
        width="stretch", hide_index=True,
    )

    if st.button("Gerar matriz de similaridade completa"):
        S = cosine_similarity(embeddings)
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(S, xticklabels=df["id"], yticklabels=df["id"], cmap="viridis", square=True,
                    cbar_kws={"label": "cosseno"}, ax=ax)
        ax.tick_params(labelsize=6)
        ax.set_title(f"Similaridade de cosseno entre as {len(df)} manifestações — {nome_modelo_exibicao}")
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# 🌐 Espaço Vetorial
# ----------------------------------------------------------------------------
with aba_espaco:
    st.subheader("Visualização 2D do espaço semântico")
    metodo = st.radio("Método de redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)
    coords = reduzir_dimensionalidade(embeddings, metodo)

    fig, ax = plt.subplots(figsize=(8, 6))
    categorias = sorted(df["categoria_oficial"].unique())
    cores = dict(zip(categorias, plt.cm.tab10.colors))
    for cat in categorias:
        mask = (df["categoria_oficial"] == cat).values
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cat, color=cores[cat], s=60, edgecolor="black")
    for (x, y), mid in zip(coords, df["id"]):
        ax.annotate(mid, (x, y), fontsize=6, alpha=0.7)
    ax.set_title(f"{metodo} dos embeddings ({nome_modelo_exibicao}), coloridos por categoria oficial")
    ax.legend(fontsize=8, loc="best")
    st.pyplot(fig)

    st.markdown(
        "**Os clusters semânticos coincidem com as categorias oficiais?** Avalie visualmente: se pontos de "
        "uma mesma cor formam grupos bem separados dos demais, o agrupamento por similaridade textual "
        "reforça a categorização oficial. Sobreposições costumam aparecer entre categorias correlatas na "
        "linguagem cotidiana do cidadão (por exemplo, *segurança* e *infraestrutura* quando a denúncia fala "
        "de iluminação pública, já que a falta de luz é citada tanto como causa de assaltos quanto como "
        "problema de manutenção urbana)."
    )

# ----------------------------------------------------------------------------
# 🧩 Chunking
# ----------------------------------------------------------------------------
with aba_chunking:
    st.subheader("Chunking de manifestações longas")
    texto_livre = st.text_area(
        "Cole uma manifestação longa",
        value=df.loc[df["n_caracteres"].idxmax(), "texto"],
        height=160,
    )

    col1, col2 = st.columns(2)
    chunk_size = col1.slider("chunk_size", min_value=50, max_value=800, value=400, step=10)
    chunk_overlap = col2.slider("chunk_overlap", min_value=0, max_value=300, value=100, step=10)

    if chunk_overlap >= chunk_size:
        st.error("chunk_overlap deve ser menor que chunk_size.")
    else:
        chunks = dividir_em_chunks(texto_livre, chunk_size, chunk_overlap)
        st.caption(f"{len(chunks)} chunk(s) gerado(s) a partir de {len(texto_livre)} caracteres.")

        for i, c in enumerate(chunks):
            with st.expander(f"Chunk {i} — {len(c)} caracteres"):
                st.write(c)

        if chunks:
            emb_chunks = modelo.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
            if len(chunks) >= 2:
                coords_chunks = PCA(n_components=2, random_state=42).fit_transform(emb_chunks) \
                    if len(chunks) > 2 else emb_chunks[:, :2]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(coords_chunks[:, 0], coords_chunks[:, 1], color="gray", alpha=0.5, lw=1)
                ax.scatter(coords_chunks[:, 0], coords_chunks[:, 1], s=80, edgecolor="black")
                for i, (x, y) in enumerate(coords_chunks):
                    ax.annotate(str(i), (x, y), fontsize=9, ha="center", va="center")
                ax.set_title("Embeddings dos chunks (PCA), na ordem em que aparecem no texto")
                st.pyplot(fig)
            else:
                st.info("Apenas 1 chunk gerado — reduza o chunk_size para ver a projeção 2D.")
