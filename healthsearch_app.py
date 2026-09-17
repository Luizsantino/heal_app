import re
import unicodedata
import pandas as pd
import streamlit as st

#Configuração da página streamlit

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

# Stopwords em portugês voltadas para limpeza léxica

STOPWORDS_pt = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos",
    "pelas", "com", "para", "e", "ou", "que", "se", "no", "ao", "aos", "à",
    "às", "seu", "sua", "seus", "suas", "este", "esta", "estes", "estas",
    "isso", "esse", "essa", "esses", "essas", "foi", "são", "ser", "ter", "ha"
}

# Pré-processamento e tokenização ( Fase 1)

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
    # Extrai tokens alfanuméricos preservando traços para capturar códigos médicos
    tokens = re.findall(r"\b[\w]+(?:-[\w]+)*\b", normalizado)
    # Filtra stopwords e tokens muito curtos desnecessários
    tokens_limpos = [t for t in tokens if t not in STOPWORDS_pt and len(t) > 1 ]
    return tokens_limpos

@st.cache_data
def carregar_corpus() -> pd.DataFrame:
    df = pd.DataFrame(CORPUS_DATA)
    # Gera os tokens léxicos com base na junção de título + conteúdo
    df["texto_completo"] = df["titulo"] + " - " + df["conteudo"]
    df["tokens"] = df["texto_completo"].apply(preprocess_lexical)
    return df

# Inicialização do dataframe
df_corpus = carregar_corpus()

# Cabeçalho da aplicação

st.title("🩺  HealthSearch: Motor de Busca Híbrido")
st.markdown(
    """
    **Disciplina:** Tendências em Ciência da Computação (UNIPÊ)  
    **Docente:** Me. Ricardo Roberto de Lima  
    *Demonstração prática da fusão entre recuperação léxica (Okapi BM25) e semântica vetorial (Embeddings).*
    """
)

# Exibição do Corpus em expander para conferência
with st.expander("📚 Visualizar Corpus de Diretrizes Médicas", expanded=False):
    st.dataframe(
        df_corpus[["id", "titulo", "conteudo"]],
        use_container_width=True,
        hide_index=True
    )