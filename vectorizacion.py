"""
Checkpoint 3: Vectorización de texto.
Reutiliza la limpieza del checkpoint 2 (limpieza.py) y aplica:
  1. Bolsa de palabras (Bag of Words)  -> CountVectorizer
  2. One-Hot / binario                 -> CountVectorizer(binary=True)
  3. TF-IDF                            -> TfidfVectorizer
  4. N-gramas (bigramas)               -> TfidfVectorizer(ngram_range=(1, 2))
  5. Word Embeddings                   -> Word2Vec (gensim)
Además compara párrafos con similitud coseno.
"""
import os
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from limpieza import cargar_texto, limpiar

CARPETA = "resultados"
os.makedirs(CARPETA, exist_ok=True)

# ---------------------------------------------------------------
# 0. CORPUS: cada párrafo del capítulo es un "documento"
# ---------------------------------------------------------------
texto = cargar_texto()
parrafos = [p for p in re.split(r"\n\s*\n", texto) if len(p.split()) > 5]

docs_tokens = [limpiar(p) for p in parrafos]          # listas de lemas (para Word2Vec)
corpus = [" ".join(t) for t in docs_tokens if t]       # strings limpios (para sklearn)
nombres = [f"parrafo_{i + 1}" for i in range(len(corpus))]

print(f"Documentos (párrafos) en el corpus: {len(corpus)}")


def top_terminos(matriz, vocab, n=10):
    """Suma cada columna y devuelve los n términos con mayor peso."""
    pesos = np.asarray(matriz.sum(axis=0)).ravel()
    orden = pesos.argsort()[::-1][:n]
    return [(str(vocab[i]), round(float(pesos[i]), 3)) for i in orden]


# ---------------------------------------------------------------
# 1. BOLSA DE PALABRAS (conteo de frecuencias)
# ---------------------------------------------------------------
bow = CountVectorizer()
X_bow = bow.fit_transform(corpus)
vocab_bow = bow.get_feature_names_out()

print("\n=== 1. Bag of Words ===")
print(f"Forma de la matriz: {X_bow.shape} (documentos x vocabulario)")
print(f"Densidad: {X_bow.nnz / np.prod(X_bow.shape) * 100:.2f}% de celdas distintas de 0")
print("Top 10 por frecuencia:", top_terminos(X_bow, vocab_bow))
pd.DataFrame(X_bow.toarray(), index=nombres, columns=vocab_bow).to_csv(f"{CARPETA}/bow.csv")

# ---------------------------------------------------------------
# 2. ONE-HOT / BINARIO (presencia o ausencia)
# ---------------------------------------------------------------
onehot = CountVectorizer(binary=True)
X_bin = onehot.fit_transform(corpus)

print("\n=== 2. One-Hot (binario) ===")
print("Top 10 por número de párrafos donde aparece:",
      top_terminos(X_bin, onehot.get_feature_names_out()))

# ---------------------------------------------------------------
# 3. TF-IDF (frecuencia ponderada por rareza)
# ---------------------------------------------------------------
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
vocab_tfidf = tfidf.get_feature_names_out()

print("\n=== 3. TF-IDF ===")
print(f"Forma de la matriz: {X_tfidf.shape}")
print("Top 10 por peso TF-IDF:", top_terminos(X_tfidf, vocab_tfidf))

idf = pd.Series(tfidf.idf_, index=vocab_tfidf).sort_values()
print("IDF más bajo (palabras comunes):", list(idf.head(5).round(2).items()))
print("IDF más alto (palabras raras):  ", list(idf.tail(5).round(2).items()))
pd.DataFrame(X_tfidf.toarray(), index=nombres, columns=vocab_tfidf).round(4) \
    .to_csv(f"{CARPETA}/tfidf.csv")

# Palabras más representativas de un párrafo concreto
fila = X_tfidf[0].toarray().ravel()
print("Palabras clave del párrafo 1:",
      [vocab_tfidf[i] for i in fila.argsort()[::-1][:5]])

# ---------------------------------------------------------------
# 4. N-GRAMAS (unigramas + bigramas)
# ---------------------------------------------------------------
ngramas = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_ng = ngramas.fit_transform(corpus)
vocab_ng = ngramas.get_feature_names_out()
bigramas = [t for t in vocab_ng if " " in t]

print("\n=== 4. N-gramas ===")
print(f"Forma de la matriz: {X_ng.shape} ({len(bigramas)} bigramas que aparecen en 2+ párrafos)")
idx_bi = [i for i, t in enumerate(vocab_ng) if " " in t]
print("Top 10 bigramas:", top_terminos(X_ng[:, idx_bi], np.array(bigramas)))

# ---------------------------------------------------------------
# 5. WORD EMBEDDINGS (Word2Vec)
# ---------------------------------------------------------------
w2v = Word2Vec(
    sentences=[t for t in docs_tokens if t],
    vector_size=50,   # dimensiones de cada vector
    window=5,         # palabras de contexto a cada lado
    min_count=2,      # ignora palabras que aparecen 1 vez
    sg=1,             # skip-gram (mejor para corpus pequeños)
    epochs=100,
    seed=42,
    workers=1,
)
w2v.save(f"{CARPETA}/word2vec.model")

print("\n=== 5. Word2Vec ===")
print(f"Vocabulario: {len(w2v.wv)} palabras, vectores de {w2v.vector_size} dimensiones")
print("Primeros 5 valores del vector 'hábito':", w2v.wv["hábito"][:5].round(3))
for palabra in ["hábito", "meta", "sistema"]:
    if palabra in w2v.wv:
        similares = [(p, round(s, 2)) for p, s in w2v.wv.most_similar(palabra, topn=5)]
        print(f"Más parecidas a '{palabra}':", similares)

# ---------------------------------------------------------------
# 6. SIMILITUD COSENO ENTRE PÁRRAFOS (sobre TF-IDF)
# ---------------------------------------------------------------
sim = cosine_similarity(X_tfidf)
np.fill_diagonal(sim, 0)
i, j = np.unravel_index(sim.argmax(), sim.shape)

print("\n=== 6. Similitud coseno (TF-IDF) ===")
print(f"Párrafos más parecidos: {nombres[i]} y {nombres[j]} (similitud {sim[i, j]:.2f})")
print("  ->", corpus[i][:90], "...")
print("  ->", corpus[j][:90], "...")
pd.DataFrame(sim, index=nombres, columns=nombres).round(3) \
    .to_csv(f"{CARPETA}/similitud_coseno.csv")

print(f"\nMatrices y modelo guardados en la carpeta '{CARPETA}/'")
