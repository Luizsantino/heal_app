import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página streamlit
st.set_page_config(
    page_title="HealthSearch - Motor Híbrido",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fase 1: Corpus Médico
CORPUS_DATA = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardioagrama CÒD-ECG-12D em até 10 minutos."
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": "O usuo imediato de ácido acetilsalicílico e antiagragantes plaquetários reduz a mortalidade no infarto agudo do miocádio."
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": "A crise hipertensiva severa requer administração anti-hitensivos venosos e monitoramento contínuo da pressão arterial na UTI."
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."
    }
]

# Stopwords em português voltadas para limpeza léxica
STOPWORDS_pt = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos",
    "pelas", "com", "para", "e", "ou", "que", "se", "ao", "aos", "à",
    "às", "seu", "sua", "seus", "suas", "este", "esta", "estes", "estas",
    "isso", "esse", "essa", "esses", "essas", "foi", "são", "ser", "ter", "ha"
}

# Pré-processamento e tokenização (Fase 1)
def normalizar_texto(texto: str) -> str:
    """Normaliza o texto removendo acentuações e convertendo para minúsculas."""
    texto = texto.lower()
    texto_norm = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto_norm if unicodedata.category(c) != "Mn")

def preprocess_lexical(texto: str) -> list[str]:
    """
    Tokenização e limpeza especializada para o motor léxico (BM25):
    - Converte para minúsculas e normaliza acentos.
    - Preserva termos alfanuméricos com hífens (ex: cód-ecg-12d).
    - Remove stopwords em português.
    """
    normalizado = normalizar_texto(texto)
    tokens = re.findall(r"\b[\w]+(?:-[\w]+)*\b", normalizado)
    tokens_limpos = [t for t in tokens if t not in STOPWORDS_pt and len(t) > 1]
    return tokens_limpos

@st.cache_data
def carregar_corpus() -> pd.DataFrame:
    df = pd.DataFrame(CORPUS_DATA)
    df["texto_completo"] = df["titulo"] + " - " + df["conteudo"]
    df["tokens"] = df["texto_completo"].apply(preprocess_lexical)
    return df

df_corpus = carregar_corpus()

# Fase 2: Motor Léxico OKAPI BM25
def executar_busca_bm25(query: str, corpus_tokens: list[list[str]], k1: float, b: float) -> list[float]:
    """
    Executa a recuperação léxica Okapi BM25 recalculando com base em k1 e b.
    Retorna uma lista de pontuações brutas correspondente a cada documento.
    """
    tokens_consulta = preprocess_lexical(query)
    if not tokens_consulta:
        return [0.0] * len(corpus_tokens)

    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    scores = bm25.get_scores(tokens_consulta)
    return [float(s) for s in scores]

# Fase 3: Motor Semântico com Embeddings
@st.cache_resource(show_spinner="Carregando modelo de embeddings semânticos denso...")
def carregar_modelo_semantico():
    #modelo multilingue de alta precisão semântica para o português
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
modelo_semantico = carregar_modelo_semantico()

@st.cache_data
def gerar_embeddings_corpus(textos: list[str]) -> np.ndarray:
    """ Gera e faz cache dos embeddings densos do corpus."""
    return modelo_semantico.encode(textos, normalize_embeddings=True)

embeddings_corpus = gerar_embeddings_corpus(df_corpus["texto_completo"].tolist())

def executar_busca_semantica(query: str, embeddings_docs: np.ndarray) -> list[float]:
    """Calcula a similaridade de cosseno entre a consulta e os documentos."""
    if not query.strip():
        return [0.0] * len(embeddings_docs)
    query_embedding = modelo_semantico.encode([query], normalize_embeddings=True)
    # Cosine similarity entre o vetor de query (1 x D) e os docs (N x D)
    similaridades = cosine_similarity(query_embedding, embeddings_docs)[0]
    return [float(s) for s in similaridades]



# Barra Lateral (sidebar) Calibração de parâmetros BM25
with st.sidebar:
    st.header("⚙️ Calibração de Parâmetros")
    st.markdown("Ajuste a dinâmica dos motores de recuperação da informação:")

    st.subheader("1. Motor Léxico (BM25)")
    k1_param = st.slider(
        label="Parâmetro k₁ (Saturação de TF)",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Controla a rapidez com que a repetição de um termo satura a pontuação. Valores mais altos dão mais peso a frequências repetidas."
    )

    b_param = st.slider(
        label="Parâmetro b (Normalização por Tamanho)",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Controla a penalidade aplicada a documentos longos. 1.0 penaliza integralmente o tamanho; 0.0 ignora o tamanho do texto."
    )

    st.divider()
    st.caption("💡 *Dica de teste:* Pesquise por termos exatos como `CÓD-ECG-12D` para avaliar a força do BM25.")

# Cabeçalho & Área de Consulta
st.title("🩺 HealthSearch: Motor de Busca Híbrido")
st.markdown(
    """
    **Disciplina:** Tendências em Ciência da Computação (UNIPÊ)  
    **Docente:** Me. Ricardo Roberto de Lima  
    *Demonstração prática da fusão entre recuperação léxica (Okapi BM25) e semântica vetorial (Embeddings).*
    """
)

# Exibição do Corpus em expander
with st.expander("📚 Corpus Médico Indexado (6 Protocolos Clínicos)", expanded=False):
    st.dataframe(
        df_corpus[["id", "titulo", "conteudo"]],
        width="stretch",
        hide_index=True
    )

st.subheader("🔍 Painel de Pesquisa Clínica")

# Inicialização da chave no session_state
if "query_input" not in st.session_state:
    st.session_state["query_input"] = "CÓD-ECG-12D"

# Exemplos rápidos de teste prático
cols_exemplos = st.columns(3)
with cols_exemplos[0]:
    if st.button("📌 Testar: 'CÓD-ECG-12D' (Termo Exato)"):
        st.session_state["query_input"] = "CÓD-ECG-12D"
        st.rerun()
with cols_exemplos[1]:
    if st.button("📌 Testar: 'infarto e ataque cardíaco' (Sinônimo)"):
        st.session_state["query_input"] = "infarto e ataque cardíaco"
        st.rerun()
with cols_exemplos[2]:
    if st.button("📌 Testar: 'pressão alta na UTI'"):
        st.session_state["query_input"] = "pressão alta na UTI"
        st.rerun()

query = st.text_input(
    "Digite sua consulta clínica (termo exato, código médico ou descrição livre):",
    key="query_input",
    placeholder="Ex: CÓD-ECG-12D, parada cardíaca, trombolíticos..."
)

# Execução do Motor BM25 e Demonstração preliminar
if query.strip():
    # 1. Executa BM25
    scores_bm25 = executar_busca_bm25(query, df_corpus["tokens"].tolist(), k1=k1_param, b=b_param)
    
    # 2. Executa Semântico
    scores_semantico = executar_busca_semantica(query, embeddings_corpus)
    
    # Monta DataFrames de Ranking
    df_resultados = df_corpus.copy()
    df_resultados["Score_BM25"] = scores_bm25
    df_resultados["Rank_BM25"] = df_resultados["Score_BM25"].rank(ascending=False, method="min").astype(int)
    
    df_resultados["Score_Semantico"] = scores_semantico
    df_resultados["Rank_Semantico"] = df_resultados["Score_Semantico"].rank(ascending=False, method="min").astype(int)

    st.markdown("---")
    st.subheader("⚔️ Confronto: Léxico (BM25) vs. Semântico (Embeddings)")

    col_lex, col_sem = st.columns(2)

    with col_lex:
        st.markdown("#### 📖 Motor Léxico (BM25)")
        df_lex_view = df_resultados.sort_values(by="Rank_BM25")[["Rank_BM25", "id", "titulo", "Score_BM25"]]
        st.dataframe(df_lex_view, use_container_width=True, hide_index=True)
        top_lex = df_lex_view.iloc[0]
        if top_lex["Score_BM25"] > 0:
            st.success(f"**Top 1:** {top_lex['id']} ({top_lex['titulo']}) — Score: `{top_lex['Score_BM25']:.3f}`")
        else:
            st.warning("⚠️ BM25 falhou em recuperar correspondências exatas.")

    with col_sem:
        st.markdown("#### 🧠 Motor Semântico (Embeddings)")
        df_sem_view = df_resultados.sort_values(by="Rank_Semantico")[["Rank_Semantico", "id", "titulo", "Score_Semantico"]]
        st.dataframe(df_sem_view, use_container_width=True, hide_index=True)
        top_sem = df_sem_view.iloc[0]
        st.info(f"**Top 1:** {top_sem['id']} ({top_sem['titulo']}) — Cosseno: `{top_sem['Score_Semantico']:.4f}`")