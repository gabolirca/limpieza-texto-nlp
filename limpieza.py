"""
Limpieza de texto: normalización y lematización con spaCy.
Texto: Hábitos atómicos (James Clear), Capítulo 1 (fragmento con fines educativos).
"""
import re
import unicodedata
from collections import Counter

import spacy

ARCHIVO_ENTRADA = "habitos_atomicos_cap1.txt"
ARCHIVO_SALIDA = "habitos_atomicos_limpio.txt"

# ---------------------------------------------------------------
# 1. INGESTA
# ---------------------------------------------------------------
with open(ARCHIVO_ENTRADA, encoding="utf-8") as f:
    texto_original = f.read()

print(f"Caracteres originales: {len(texto_original):,}")
print(f"Palabras originales:   {len(texto_original.split()):,}")

# ---------------------------------------------------------------
# 2. PRE-LIMPIEZA (ruido propio de un PDF convertido a .txt)
# ---------------------------------------------------------------
def pre_limpieza(texto):
    # Unificar la codificación Unicode (á puede venir como a + ´)
    texto = unicodedata.normalize("NFC", texto)
    # Quitar encabezados de página repetidos
    texto = re.sub(r"(?im)^\s*(el sorprendente poder de los )?hábitos atómicos\s*$", "", texto)
    # Quitar pies de página de imprenta (ej. "HABITOS ATOMICOS-EDICION ESPECIAL-01.indd 27")
    texto = re.sub(r"(?m)^.*\.indd.*$", "", texto)
    # Quitar números de página (líneas que solo tienen un número)
    texto = re.sub(r"(?m)^\s*\d+\s*$", "", texto)
    # Unir palabras cortadas con guion al final de línea: "pala-\nbra" -> "palabra"
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    # Saltos de página (form feed)
    texto = texto.replace("\f", "\n")
    return texto


# ---------------------------------------------------------------
# 3. NORMALIZACIÓN
# ---------------------------------------------------------------
def normalizar(texto):
    texto = texto.lower()                               # minúsculas
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)  # URLs
    texto = re.sub(r"\S+@\S+", " ", texto)              # correos
    texto = re.sub(r"\d+", " ", texto)                  # números
    texto = re.sub(r"[^\wáéíóúüñ\s]", " ", texto)       # símbolos y puntuación
    texto = re.sub(r"\s+", " ", texto).strip()          # espacios múltiples y saltos
    return texto


# ---------------------------------------------------------------
# 4. TOKENIZACIÓN, STOPWORDS Y LEMATIZACIÓN (spaCy)
# ---------------------------------------------------------------
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])


def lematizar(texto):
    doc = nlp(texto)
    lemas = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if not token.is_alpha or len(token) < 3:
            continue
        lemas.append(token.lemma_.lower())
    return lemas


# ---------------------------------------------------------------
# EJECUCIÓN DEL PROCESO
# ---------------------------------------------------------------
texto_pre = pre_limpieza(texto_original)
texto_norm = normalizar(texto_pre)
lemas = lematizar(texto_norm)
texto_limpio = " ".join(lemas)

with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
    f.write(texto_limpio)

# ---------------------------------------------------------------
# RESULTADOS
# ---------------------------------------------------------------
print(f"\nTokens tras limpieza:  {len(lemas):,}")
print(f"Vocabulario único:     {len(set(lemas)):,}")
print(f"Reducción de palabras: {100 - len(lemas) / len(texto_original.split()) * 100:.1f}%")

print("\nEjemplo de normalización y lematización:")
ejemplo = "Los hábitos pequeños generan cambios enormes cuando se repiten."
print("  Original:", ejemplo)
print("  Limpio:  ", " ".join(lematizar(normalizar(ejemplo))))

print("\nTop 20 lemas más frecuentes:")
for palabra, freq in Counter(lemas).most_common(20):
    print(f"  {palabra:<15} {freq}")

print(f"\nTexto limpio guardado en {ARCHIVO_SALIDA}")
