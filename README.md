# 🎬 Mini RAG System — Movie Plot Question Answering
A lightweight Retrieval-Augmented Generation (RAG) system built for the take-home assignment.

This repo contains:
- `mini_rag.ipynb` — main Colab/local notebook (end-to-end demo)
- `main.py` — CLI runner to build the RAG index and answer a query
- `requirements.txt` — Python dependencies
- `data/wiki_movie_plots_deduped.csv` — Download the dataset from https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?resource=download&utm_source=chatgpt.com and place it in folder (user should add CSV with Title,Plot)

---

## Quickstart (Colab)
1. Open `mini_rag.ipynb` in Colab.
2. Run the install cell to install dependencies.
3. Upload `wiki_movie_plots_deduped.csv` (200–500 rows, columns: Title,Plot).
4. Set `GENAI_MOCK_MODE = True` for demo without API calls, or set `GOOGLE_API_KEY` and `GENAI_MOCK_MODE = False` for real Gemini calls.
5. Run the demo cells (build index, run query, view JSON output).

## Quickstart (Local)
1. Create and activate a venv (Python 3.10/3.11 recommended):
   ```bash
   python -m venv rag-env
   source rag-env/bin/activate   # macOS/Linux
   rag-env\\Scripts\\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install --index-url https://download.pytorch.org/whl/cpu/ torch torchvision
   pip install -r requirements.txt
   ```
3. Add your CSV to `data/wiki_movie_plots_deduped.csv`.
4. Run the CLI:
   ```bash
   python main.py --csv data/wiki_movie_plots_deduped.csv --query "Which movie features an evil artificial intelligence?" --mock
   ```

## Files
- `mini_rag.ipynb` — Notebook (Colab-friendly). Use this for the demo and Loom recording.
- `main.py` — CLI runner that loads data, builds embeddings & FAISS, retrieves, and prints JSON output.
- `requirements.txt` — dependencies required to run.
- `data/wiki_movie_plots_deduped.csv` — add the dataset here (not included).

---
