"""
Checkpoint 4: Vectorización con semántica distribucional.
"Una palabra se conoce por las palabras que la acompañan" (Firth, 1957).

Dos enfoques:
  A. Basado en conteo:   matriz de co-ocurrencia -> PPMI -> SVD (vectores densos)
  B. Basado en predicción: Word2Vec (skip-gram), entrenado en vectorizacion.py
Y su representación gráfica:
  imagenes/01_word2vec_pca.png        espacio de palabras Word2Vec (PCA 2D)
  imagenes/02_ppmi_svd.png            espacio de palabras PPMI + SVD (2D)
  imagenes/03_word2vec_tsne.png       espacio de palabras Word2Vec (t-SNE 2D)
  imagenes/04_parrafos_lsa.png        espacio de documentos (LSA) con clusters
  imagenes/05_similitud_palabras.png  mapa de calor de similitud coseno
"""
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from limpieza import cargar_texto, limpiar

os.makedirs("imagenes", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
SEMILLA = 42
N_PALABRAS = 40          # palabras más frecuentes que se grafican
VENTANA = 4              # contexto a cada lado para la co-ocurrencia

# ---------------------------------------------------------------
# 1. CARGA Y LIMPIEZA
# ---------------------------------------------------------------
texto = cargar_texto()
parrafos = [p for p in re.split(r"\n\s*\n", texto) if len(p.split()) > 5]
docs = [t for t in (limpiar(p) for p in parrafos) if t]
todos = [w for d in docs for w in d]
frecuentes = [w for w, _ in Counter(todos).most_common(N_PALABRAS)]
print(f"Párrafos: {len(docs)} | Tokens: {len(todos)} | Palabras a graficar: {N_PALABRAS}")

# ---------------------------------------------------------------
# 2A. CONTEO: MATRIZ DE CO-OCURRENCIA
# ---------------------------------------------------------------
vocab = sorted(set(todos))
idx = {w: i for i, w in enumerate(vocab)}
cooc = np.zeros((len(vocab), len(vocab)))
for d in docs:
    for i, w in enumerate(d):
        for j in range(max(0, i - VENTANA), min(len(d), i + VENTANA + 1)):
            if i != j:
                cooc[idx[w], idx[d[j]]] += 1

print(f"\nMatriz de co-ocurrencia: {cooc.shape}")
sub = pd.DataFrame(cooc, index=vocab, columns=vocab).loc[frecuentes[:10], frecuentes[:10]]
print(sub.astype(int).to_string())

# ---------------------------------------------------------------
# 2B. PPMI (Positive Pointwise Mutual Information)
#     Mide si dos palabras aparecen juntas más de lo esperado por azar
# ---------------------------------------------------------------
total = cooc.sum()
p_w = cooc.sum(axis=1, keepdims=True) / total
p_c = cooc.sum(axis=0, keepdims=True) / total
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((cooc / total) / (p_w * p_c))
ppmi = np.nan_to_num(np.maximum(pmi, 0))

# ---------------------------------------------------------------
# 2C. SVD: de vectores dispersos (|V| dimensiones) a densos (50)
# ---------------------------------------------------------------
svd = TruncatedSVD(n_components=50, random_state=SEMILLA)
vec_ppmi = svd.fit_transform(ppmi)
print(f"\nPPMI + SVD: {ppmi.shape} -> {vec_ppmi.shape} "
      f"(varianza explicada {svd.explained_variance_ratio_.sum() * 100:.1f}%)")


def vecinos_ppmi(palabra, n=5):
    sims = cosine_similarity(vec_ppmi[[idx[palabra]]], vec_ppmi).ravel()
    orden = [i for i in sims.argsort()[::-1] if vocab[i] != palabra][:n]
    return [(vocab[i], round(float(sims[i]), 2)) for i in orden]


# ---------------------------------------------------------------
# 3. PREDICCIÓN: WORD2VEC
# ---------------------------------------------------------------
ruta_modelo = "resultados/word2vec.model"
if os.path.exists(ruta_modelo):
    w2v = Word2Vec.load(ruta_modelo)
else:
    w2v = Word2Vec(docs, vector_size=50, window=5, min_count=2, sg=1,
                   epochs=100, seed=SEMILLA, workers=1)
    w2v.save(ruta_modelo)

print("\nComparación de vecinos semánticos (conteo vs predicción):")
for palabra in ["hábito", "meta", "sistema", "pequeño"]:
    print(f"  {palabra}")
    print(f"    PPMI+SVD : {vecinos_ppmi(palabra)}")
    print(f"    Word2Vec : {[(p, round(s, 2)) for p, s in w2v.wv.most_similar(palabra, topn=5)]}")


# ---------------------------------------------------------------
# 4. GRÁFICAS DE LOS ESPACIOS VECTORIALES
# ---------------------------------------------------------------
def graficar_palabras(coords, palabras, titulo, archivo, subtitulo):
    plt.figure(figsize=(11, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=40, color="#2b6cb0", alpha=0.7)
    for (x, y), w in zip(coords, palabras):
        plt.annotate(w, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=9)
    plt.title(titulo, fontsize=14, weight="bold")
    plt.suptitle(subtitulo, y=0.02, fontsize=9, color="gray")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"imagenes/{archivo}", dpi=150)
    plt.close()
    print(f"  imagenes/{archivo}")


print("\nGenerando imágenes:")
palabras_w2v = [w for w in frecuentes if w in w2v.wv]
M_w2v = np.array([w2v.wv[w] for w in palabras_w2v])

# 4.1 Word2Vec + PCA
pca = PCA(n_components=2, random_state=SEMILLA)
graficar_palabras(
    pca.fit_transform(M_w2v), palabras_w2v,
    "Espacio semántico Word2Vec (PCA)", "01_word2vec_pca.png",
    f"{len(palabras_w2v)} palabras más frecuentes · 50 dimensiones reducidas a 2 · "
    f"varianza explicada {pca.explained_variance_ratio_.sum() * 100:.1f}%")

# 4.2 PPMI + SVD
M_ppmi = vec_ppmi[[idx[w] for w in frecuentes]]
graficar_palabras(
    PCA(n_components=2, random_state=SEMILLA).fit_transform(M_ppmi), frecuentes,
    "Espacio distribucional por conteo (PPMI + SVD)", "02_ppmi_svd.png",
    f"Co-ocurrencia con ventana ±{VENTANA} · PPMI · SVD a 50 dims · PCA a 2")

# 4.3 Word2Vec + t-SNE (preserva vecindarios locales)
tsne = TSNE(n_components=2, perplexity=10, random_state=SEMILLA, init="pca")
graficar_palabras(
    tsne.fit_transform(M_w2v), palabras_w2v,
    "Espacio semántico Word2Vec (t-SNE)", "03_word2vec_tsne.png",
    "t-SNE agrupa palabras que comparten contexto; las distancias globales no son interpretables")

# 4.4 Espacio de documentos: LSA (TF-IDF + SVD) con clusters K-Means
corpus = [" ".join(d) for d in docs]
X = TfidfVectorizer().fit_transform(corpus)
lsa = TruncatedSVD(n_components=2, random_state=SEMILLA).fit_transform(X)
k = 3
grupos = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(X)

plt.figure(figsize=(11, 8))
colores = ["#2b6cb0", "#dd6b20", "#38a169"]
for g in range(k):
    m = grupos == g
    # palabras que mejor describen al grupo
    centro = np.asarray(X[m].mean(axis=0)).ravel()
    vocab_t = TfidfVectorizer().fit(corpus).get_feature_names_out()
    etiqueta = ", ".join(vocab_t[centro.argsort()[::-1][:3]])
    plt.scatter(lsa[m, 0], lsa[m, 1], s=70, color=colores[g], alpha=0.8,
                label=f"Grupo {g + 1}: {etiqueta}")
for i, (x, y) in enumerate(lsa):
    plt.annotate(str(i + 1), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
plt.title("Espacio de párrafos (LSA: TF-IDF + SVD) y clusters K-Means", fontsize=14, weight="bold")
plt.xlabel("Tema latente 1")
plt.ylabel("Tema latente 2")
plt.legend(loc="best", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("imagenes/04_parrafos_lsa.png", dpi=150)
plt.close()
print("  imagenes/04_parrafos_lsa.png")

# 4.5 Mapa de calor de similitud coseno entre palabras (Word2Vec)
top = palabras_w2v[:15]
S = cosine_similarity(np.array([w2v.wv[w] for w in top]))
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(top)), top, rotation=45, ha="right")
ax.set_yticks(range(len(top)), top)
for i in range(len(top)):
    for j in range(len(top)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if S[i, j] > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_title("Similitud semántica entre las 15 palabras más frecuentes (Word2Vec)",
             fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("imagenes/05_similitud_palabras.png", dpi=150)
plt.close()
print("  imagenes/05_similitud_palabras.png")

# Guardar vectores para reutilizarlos
pd.DataFrame(M_w2v, index=palabras_w2v).round(4).to_csv("resultados/vectores_word2vec.csv")
pd.DataFrame(M_ppmi, index=frecuentes).round(4).to_csv("resultados/vectores_ppmi_svd.csv")
print("\nVectores guardados en resultados/")
