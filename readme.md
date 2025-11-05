# RAG con Pinecone + LangChain + Gemini

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![OS](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows)

![RAG](https://img.shields.io/badge/Pattern-Retrieval--Augmented%20Generation-7957D5)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C6B72)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00A896?logo=pinecone)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-4285F4?logo=google)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-multilingual--e5-FFAE00?logo=huggingface)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)
![PDF](https://img.shields.io/badge/PDF-pypdf-2C3E50)

## Descripción
- Este proyecto implementa un pipeline RAG para consultar cualquier PDF (en este caso solo texto) con embeddings de Hugging Face, almacenamiento y recuperación en Pinecone, compresión contextual (reranking) y generación con Gemini (Google Generative AI).
- El notebook `03_rag_pinecone_gemini.ipynb` contiene los bloques ejecutables del flujo (1–8).

## Cómo Funciona (resumen)
- Extracción y partición: se extrae texto por página con `pypdf` y se generan chunks con `RecursiveCharacterTextSplitter`.
- Embeddings: se vectorizan los documentos y la consulta con `HuggingFaceEmbeddings` (`intfloat/multilingual-e5-base`).
- Índice vectorial: se crea/usa un índice Pinecone (serverless) alineado con la dimensión de los embeddings.
- Upsert: se suben los chunks con metadatos (`source`, `page`) al `namespace` activo.
- Recuperación MMR: el retriever de LangChain consulta Pinecone (`index.query`) y devuelve candidatos diversos y relevantes.
- Compresor manual: se reordena/filtra localmente por similitud coseno con un umbral y `top_k`.
- Generación: se construye el contexto con citas `[p. N]` y se invoca a Gemini con un `SYSTEM_PROMPT` claro usando LCEL.

## Pila Tecnológica
- `LangChain` (LCEL) — orquestación declarativa de la cadena RAG.
- `Pinecone` — almacenamiento y recuperación vectorial escalable (serverless AWS `us-east-1`).
- `Gemini 2.0 Flash` — LLM para responder con base en el contexto recuperado.
- `HuggingFaceEmbeddings` — modelo `multilingual-e5-base` para vectorizar texto y consultas.
- `pypdf`, `langchain-text-splitters` — extracción y chunking de PDF.
- `NumPy` — utilidades de similitud coseno en el compresor.
- `python-dotenv` — carga de variables desde `.env`.

## Variables de Entorno
- `GOOGLE_API_KEY` — clave para Gemini.
- `PINECONE_API_KEY` — clave para Pinecone.
- `PINECONE_REGION` — región (por defecto `us-east-1`).
- `PC_INDEX_NAME` — nombre del índice Pinecone (ej. `prueba`).
- `PC_NAMESPACE` — namespace; si no se define, se usa `default` o el vacío según el código.
- Opcionales: `PDF_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `PC_RESET_IF_DIM_MISMATCH`.

> [!IMPORTANT]
> Tras cambiar variables en `.env`, reinicia el kernel/notebook y ejecuta BLOQUES 1–7 en orden para que `vectorstore/retriever/cretriever` adopten el nuevo índice y namespace.

> [!NOTE]
> Si prefieres `INDEX_NAME` en `.env`, ajusta el código para leer `INDEX_NAME = os.getenv("INDEX_NAME", "prueba")` o agrega `PC_INDEX_NAME` en el `.env`.

## Instalación
```bash
pip install -U langchain langchain-pinecone langchain-huggingface langchain-google-genai langchain-text-splitters pinecone pypdf python-dotenv numpy jupyter
```

> [!TIP]
> Usa un entorno virtual para aislar dependencias: `python -m venv .venv` y actívalo (`.venv\Scripts\Activate.ps1` en Windows PowerShell).

## Ejecución (orden recomendado)
1. BLOQUE 1 — Configuración y entorno.
2. BLOQUE 2 — Embeddings locales (Hugging Face).
3. BLOQUE 3 — Ingesta de PDF y chunking.
4. BLOQUE 4 — Pinecone: creación/validación del índice.
5. BLOQUE 5 — Upsert (subida de chunks a Pinecone).
6. BLOQUE 6 — Retriever MMR y compresión contextual.
7. BLOQUE 7 — Cadena RAG (LCEL) con Gemini.
8. BLOQUE 8 — Consulta de ejemplo y páginas citadas.

> [!TIP]
> Si saltas el BLOQUE 5, el BLOQUE 6 reconecta `PineconeVectorStore` al índice/namespace activo para poder consultar; si el namespace no tiene datos, verás `pages = []`.

## Verificar que consultas Pinecone
```python
stats = index.describe_index_stats()
ns = NAMESPACE or ""
vc = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
print(f"Namespace activo: '{ns}' · vector_count={vc}")
```
- Cambia `PC_NAMESPACE` a uno vacío o nuevo, reinicia y ejecuta BLOQUES 1–4 y 6–8 (sin upsert): deberías ver `pages = []`.

> [!WARNING]
> Si el namespace activo no contiene vectores (no hiciste upsert), `docs_used` estará vacío y `pages = []`. El modelo debería indicar falta de información según el `SYSTEM_PROMPT`.

> [!IMPORTANT]
> La dimensión del índice debe coincidir con `detected_dim` del modelo de embeddings. Si hay discrepancia y defines `PC_RESET_IF_DIM_MISMATCH=true`, el índice se recreará para alinearse.

## Ajustes Rápidos
- Chunking: `CHUNK_SIZE` y `CHUNK_OVERLAP` para equilibrio de contexto y redundancia.
- MMR: `k`, `fetch_k`, `lambda_mult` para diversidad vs relevancia.
- Compresor: `similarity_threshold` y `top_k` para controlar el filtro.
- Prompt: reforzar “no inventar” y exigir citas en `[p. N]`.

> [!NOTE]
> En consultas multilingües, añade sinónimos en la pregunta (por ejemplo: "conclusión", "conclusiones", "conclusion") para mejorar el recall.

## Seguridad
- Mantén tus claves en `.env` y evita exponerlas en notebooks.
- Si compartiste claves, obviamente debes rotarlas, por si las dudas lo pongo, nunca expongan el `.env`.

> [!CAUTION]
> No subas `.env` al repositorio ni compartas claves en issues/PRs/notebooks. Evita incluir credenciales en capturas o logs.

> [!IMPORTANT]
> Si alguna clave se expuso, rótala de inmediato en Google y Pinecone y actualiza tu `.env` antes de continuar. Son cosas obvias pero todos empezamos de 0 alguna vez.
