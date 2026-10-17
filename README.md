# Limpieza de texto con spaCy — Hábitos atómicos (Cap. 1)

Proyecto de Procesamiento de Datos: normalización y lematización de un libro en español.

## Contenido
- `habitos_atomicos_cap1.txt` — texto a procesar (fragmento: Capítulo 1 de *Hábitos atómicos*, James Clear, usado con fines educativos).
- `limpieza.py` — proceso de limpieza: ingesta, pre-limpieza del PDF, normalización, eliminación de stopwords y lematización.
- `requirements.txt` — dependencias congeladas con `pip freeze`.
- `habitos_atomicos_limpio.txt` — resultado generado.

## Proceso
1. **Ingesta:** lectura del `.txt` en UTF-8.
2. **Pre-limpieza:** normalización Unicode (NFC), eliminación de encabezados y números de página, unión de palabras cortadas con guion.
3. **Normalización:** minúsculas, eliminación de URLs, correos, números, puntuación y espacios extra.
4. **Lematización:** tokenización con `es_core_news_sm`, eliminación de stopwords y tokens cortos, reducción a lema.

## Ejecución
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python limpieza.py
```
