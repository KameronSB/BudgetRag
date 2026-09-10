# Animal Facts RAG

A small, local Retrieval-Augmented Generation (RAG) project that answers questions about animals using a CSV dataset, FAISS vector search, and Ollama-hosted language models.

This project is intended as a simple learning example for how a RAG pipeline works end to end:

1. Load animal facts from a CSV file.
2. Convert each fact into an embedding.
3. Store the embeddings in a FAISS vector index.
4. Retrieve the most relevant facts for a user question.
5. Ask a local language model to answer using the retrieved context.

## Companion Blog Post

I wrote about this project and the ideas behind it here:

[Read the blog post](https://www.kameronbains.com/view_post/68d10edf2e3f810b2c3a2540)

## Tech Stack

- Python
- Pandas
- NumPy
- FAISS
- Ollama
- `embeddinggemma` for embeddings
- `smollm:360m` for response generation

## Project Structure

```text
.
|-- AnimalFacts.csv
|-- animal-fun-facts-dataset.csv
|-- anaimal_facts.index
|-- build_faiss_index.py
|-- createDataset.ipynb
|-- rag.py
`-- requirements.txt
```

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then pull the models used by this project:

```bash
ollama pull embeddinggemma
ollama pull smollm:360m
```

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

## Run The RAG Demo

Start the interactive question-answering script:

```bash
python rag.py
```

You should see a prompt like this:

```text
RAG ready. Ask about animals! Type 'stop' to exit.
Q>
```

Ask a question:

```text
Q> what do aardvarks eat?
```

To exit the demo, type:

```text
Q> stop
```

You can also use `exit` or `quit`.

## Rebuild The FAISS Index

The repository includes a prebuilt FAISS index:

```text
anaimal_facts.index
```

If you update `AnimalFacts.csv`, rebuild the index so the vector search uses the latest data:

```bash
python build_faiss_index.py
```

Then run the RAG demo again:

```bash
python rag.py
```

## How It Works

`build_faiss_index.py` reads the animal facts from `AnimalFacts.csv`, sends each fact to Ollama's `embeddinggemma` model, and stores the resulting vectors in a FAISS index.

`rag.py` takes a user question, embeds it with the same embedding model, searches the FAISS index for similar facts, and passes those facts as context to `smollm:360m` to generate an answer.

## Notes

- This is an educational project, not a production RAG system.
- The quality of answers depends on the dataset, embedding model, retrieval results, and generation model.
- The project currently expects the FAISS index file to be named `anaimal_facts.index`.
- The answer generation prompt asks the model to use only the retrieved context.

## Troubleshooting

If Ollama cannot connect, make sure it is running:

```bash
ollama serve
```

If a model is missing, pull it:

```bash
ollama pull embeddinggemma
ollama pull smollm:360m
```

If the index cannot be loaded, rebuild it:

```bash
python build_faiss_index.py
```

If the answers seem unrelated, rebuild the index and confirm that `AnimalFacts.csv` contains an `animal_fact` column.

## License

No license is needed, use how ever you want =)
