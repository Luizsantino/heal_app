import streamlit as st
import math
import re
import unicodedata
import pandas as pd
import nltk
from nltk.stem.rslp import RSLPStemmer

# Download de dados do NLTK

try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('rslp')

#---
# 1. BASE DE DADOS
#---

DOCUMENTS = {
   1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico."
}

STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "com",
    "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e",
    "é", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "éramos", "essa", "essas",
    "esse", "esses", "esta", "está", "estamos", "estão", "estas", "este", "estes", "estou",
    "eu", "foi", "fomos", "foram", "ha", "há", "isso", "isto", "já", "lhe", "lhes", "mais",
    "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem",
    "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "pode", "por", "qual", "quando", "que", "quem", "são",
    "se", "seja", "sem", "ser", "seu", "seus", "só", "sua", "suas", "também", "te", "tem",
    "temos", "ter", "um", "uma", "você", "vocês"
}

stemmer = RSLPStemmer()

#---
# 2. PIPELINE DE PRÉ-PROCESSAMENTO
#---

def remove_accents(text: str) -> str:
    """
    Remove acentos de uma string.
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess_text(text: str, use_stopwords: bool = True, use_stemming: bool = True) -> list[str]:
    """
    Pipeline completo de pré-processamento de texto.
    """

    text = text.lower()

    text = remove_accents(text)

    tokens = re.findall(r'\b\w+\b',text)

    if use_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS_PT]

    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens

#---
# 3. CONSTRUÇÃO DO ÍNDICE INVERTIDO
#---

def build_inverted_index(docs: dict[int, str], use_stopwords: bool, use_stemming: bool) -> dict[str, set[int]]:
    """
    Gera o Índice Invertido: Termo -> Conjunto de IDs dos Documentos
    """
    inverted_index = {}
    for doc_id, text in docs.items():
        tokens = preprocess_text(text, use_stopwords, use_stemming)
        for token in set(tokens):
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)
    return inverted_index

#---
# 4. TF-IDF E SIMILARIDADE DE COSSENO
#---

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Calcula a frequência de termo relativa ao tamnanho do documento (TF).
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {}

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    return {token: count / total_tokens for token, count in counts.items()}

def compute_idf(docs_processed: dict[int, list[str]]) -> dict[str, float]:
    """
    Calcula o Inverse Document Frequency com suavização logarítmica (base 10).
    """
    num_docs = len(docs_processed)
    all_tokens = set(token for tokens in docs_processed.values() for token in tokens)

    idf = {}
    for token in all_tokens:
        doc_count = sum(1 for tokens in docs_processed.values() if token in tokens)

        idf[token] = math.log10(num_docs / doc_count) if doc_count > 0 else 0.0
    return idf

def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """
    Calcula a Similaridade de Cosseno entre dois vetores TF-IDF (Desafio Bônus).
    """
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
