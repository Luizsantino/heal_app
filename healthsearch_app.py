import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
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

# Fase 4: Fusão RRF (Reciproval Rank Fusion)

def calcular_rrf(rank_bm25: pd.Series, rank_semantico: pd.Series, alpha: float, k_rrf: int = 60) -> pd.Series:
    """
    Score_RRF(D) = alpha * [1 / (k_rrf + Rank_BM25)] + (1 - alpha) * [1 / (k_rrf + Rank_Semantico)]
    """
    termo_bm25 = alpha * (1.0 / (k_rrf + rank_bm25))
    termo_semantico = (1.0 - alpha) * (1.0 / (k_rrf + rank_semantico))
    return termo_bm25 + termo_semantico

# Cross-encoder RE_RANKING
@st.cache_resource(show_spinner="Carregando modelo Cross-Encoder Re-Ranker...")
def carregar_modelo_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def executar_reranking_cross_encoder(query: str, df_top3: pd.DataFrame) -> pd.DataFrame:
    """ Aplca atenção cruzada total entre a query e os Top-3 do RRF."""
    ce_model = carregar_modelo_cross_encoder()
    pares = [(query, texto) for texto in df_top3["texto_completo"].tolist()]
    scores_ce = ce_model.predict(pares)

    df_reranked = df_top3.copy()
    df_reranked["Score_CrossEncoder"] = [float(s) for s in scores_ce]
    # Reoderna pela nota do Cross-Encoder
    df_reranked = df_reranked.sort_values(by="Score_CrossEncoder", ascending=False).reset_index(drop=True)
    df_reranked["Rank_Final_CE"] = [1, 2, 3]
    return df_reranked

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

    st.subheader("2. Fusão Híbrida RRF")
    alpha_param = st.slider(
        "Peso α (Balanceador)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="α = 1.0 (Apenas BM25) | α = 0.0 (Apenas Semântico) | α = 0.50 (Equilibrado)"
    )
    k_rrf_param = 60 # Valor fixo de k para RRF, conforme literatura
    st.caption(f"Constante de suavização fixa: **k_rrf = {k_rrf_param}**")
    
    st.divider()
    st.subheader("3. Cross-Encoder Re-Ranking (Opcional)")
    ativar_cross_encoder = st.checkbox(
        "Ativar Cross-Encoder Re-Ranking (Top-3)",
        value=False,
        help="Se ativado, os 3 melhores resultados do RRF serão reordenados usando atenção cruzada total entre a consulta e os documentos."
    )
    st.markdown("### 🎯 Guia Rápido de Testes")
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

# Processamento dos Motores & Fusão
if query.strip():
    # 1. Executa BM25
    scores_bm25 = executar_busca_bm25(query, df_corpus["tokens"].tolist(), k1=k1_param, b=b_param)
    
    # 2. Executa Semântico
    scores_semantico = executar_busca_semantica(query, embeddings_corpus)
    
    # 3. Consolidação de Ranks
    df_resultados = df_corpus.copy()
    df_resultados["Score_BM25"] = scores_bm25
    df_resultados["Rank_BM25"] = df_resultados["Score_BM25"].rank(ascending=False, method="min").astype(int)
    
    df_resultados["Score_Semantico"] = scores_semantico
    df_resultados["Rank_Semantico"] = df_resultados["Score_Semantico"].rank(ascending=False, method="min").astype(int)

    # 4. Cálculo RRF
    # 4. Cálculo RRF
    df_resultados["Score_RRF"] = calcular_rrf(df_resultados["Rank_BM25"], df_resultados["Rank_Semantico"], alpha=alpha_param, k_rrf=k_rrf_param)
    df_resultados["Rank_RRF"] = df_resultados["Score_RRF"].rank(ascending=False, method="min").astype(int)



    st.markdown("---")
    # ==========================================================================
    # ABAS DA INTERFACE (CRITÉRIO OBRIGATÓRIO DE AVALIAÇÃO)
    # ==========================================================================
    tab_hibrido, tab_lexico, tab_semantico, tab_matriz = st.tabs([
        "🧬 1. Híbrido RRF (Fusão)",
        "📖 2. Motor Léxico (BM25)",
        "🧠 3. Motor Semântico (Embeddings)",
        "📊 4. Matriz Comparativa & Re-Ranking"
        
    ])

    # --------------------------------------------------------------------------
    # ABA 1: HÍBRIDO RRF
    # --------------------------------------------------------------------------
    with tab_hibrido:
        st.markdown(f"### 🏆 Resultados Unificados (RRF com α = {alpha_param:.2f})")
        st.caption(f"Fórmula: $\\text{{Score}}_{{RRF}}(D) = {alpha_param:.2f} \\cdot \\frac{{1}}{{60 + \\text{{Rank}}_{{BM25}}}} + {(1-alpha_param):.2f} \\cdot \\frac{{1}}{{60 + \\text{{Rank}}_{{Semantico}}}}$")
        
        df_hibrido_ordenado = df_resultados.sort_values(by=["Rank_RRF", "Score_RRF"], ascending=[True, False]).reset_index(drop=True)
        # Bloco do Bônus: Cross-Encoder aplicado aos Top-3
        if ativar_cross_encoder:
            st.success("⭐ **Camada Bônus Ativa: Cross-Encoder Re-Ranking aplicada aos Top-3!**")
            df_top3 = df_resultados.sort_values(by=["Rank_RRF"], ascending=True).head(3).copy()
            df_top3_reranked = executar_reranking_cross_encoder(query, df_top3)


        for idx, row in df_hibrido_ordenado.iterrows():
            with st.container():
                col_rank, col_conteudo, col_metricas = st.columns([1, 4, 2])
                with col_rank:
                    st.metric("Posição RRF", f"#{row['Rank_RRF']}")
                with col_conteudo:
                    st.markdown(f"#### {row['id']} — {row['titulo']}")
                    st.write(row['conteudo'])
                with col_metricas:
                    st.caption(f"**Score RRF:** `{row['Score_RRF']:.6f}`")
                    st.caption(f"**Posição BM25:** #{row['Rank_BM25']} (Score: {row['Score_BM25']:.3f})")
                    st.caption(f"**Posição Semântica:** #{row['Rank_Semantico']} (Cosseno: {row['Score_Semantico']:.4f})")
                st.divider()

    # --------------------------------------------------------------------------
    # ABA 2: LÉXICO BM25
    # --------------------------------------------------------------------------
    with tab_lexico:
        st.markdown("### 📖 Ranking Puro do Okapi BM25")
        st.caption(f"Tokens processados da consulta: `{preprocess_lexical(query)}`")
        df_lex_view = df_resultados.sort_values(by=["Rank_BM25", "id"])[["Rank_BM25", "id", "titulo", "Score_BM25", "conteudo"]]
        st.dataframe(df_lex_view, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # ABA 3: SEMÂNTICO VETORIAL
    # --------------------------------------------------------------------------
    with tab_semantico:
        st.markdown("### 🧠 Ranking Puro da Busca Semântica Vetorial")
        st.caption("Similaridade de Cosseno entre a consulta e os embeddings normalizados do corpus.")
        df_sem_view = df_resultados.sort_values(by=["Rank_Semantico", "id"])[["Rank_Semantico", "id", "titulo", "Score_Semantico", "conteudo"]]
        st.dataframe(df_sem_view, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # ABA 4: MATRIZ COMPARATIVA & DIAGNÓSTICO
    # --------------------------------------------------------------------------
    with tab_matriz:
        st.markdown("### 📊 Matriz Comparativa de Desempenho")
        
        colunas_exibir = ["id", "titulo", "Rank_BM25", "Rank_Semantico", "Rank_RRF", "Score_RRF"]
        tabela_comp = df_resultados[colunas_exibir].copy()
        
        if ativar_cross_encoder:
            df_ce_temp = executar_reranking_cross_encoder(query, df_resultados.head(3))
            map_ce_rank = dict(zip(df_ce_temp["id"], df_ce_temp["Rank_Final_CE"]))
            map_ce_score = dict(zip(df_ce_temp["id"], df_ce_temp["Score_CrossEncoder"]))
            
            tabela_comp["Rank_CrossEncoder"] = tabela_comp["id"].map(map_ce_rank).fillna("-")
            tabela_comp["Score_CrossEncoder"] = tabela_comp["id"].map(map_ce_score)
        
        st.dataframe(tabela_comp, use_container_width=True, hide_index=True)
        
        st.markdown("#### 📈 Variação de Ranks dos Candidatos")
        df_chart = df_resultados[["id", "Rank_BM25", "Rank_Semantico", "Rank_RRF"]].set_index("id")
        st.bar_chart(df_chart)
        st.caption("*Menor barra = melhor posição (#1 é o topo).*")