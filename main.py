"""
Proyecto completo: Hábitos atómicos (Cap. 1)
Carga de datos -> Limpieza -> Vectorizaciones -> Semántica distribucional
"""
import runpy

ETAPAS = [
    ("1. Carga y limpieza (normalización + lematización)", "limpieza.py"),
    ("2. Vectorización (BoW, One-Hot, TF-IDF, N-gramas, Word2Vec)", "vectorizacion.py"),
    ("3. Semántica distribucional (PPMI + SVD, Word2Vec) y gráficas", "semantica_distribucional.py"),
]

for titulo, script in ETAPAS:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    runpy.run_path(script, run_name="__main__")

print("\nProyecto completo. Revisa las carpetas 'resultados/' e 'imagenes/'.")
