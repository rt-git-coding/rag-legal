# Extracting Information from Legal Documents Using RAG

A Retrieval-Augmented Generation (RAG) pipeline over a corpus of legal agreements (NDAs, commercial contracts, merger agreements, privacy policies). The notebook covers data exploration, chunking, vector indexing, question answering with source citations, and evaluation against a benchmark.


See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pipeline is built.

## Dataset

Unzip `rag_legal.zip` to get:

| Folder | Content | Documents |
|---|---|---|
| `corpus/contractnli` | NDAs and confidentiality agreements | 93 |
| `corpus/cuad` | Commercial contracts with annotated clauses | 431 |
| `corpus/maud` | Merger and acquisition agreements | 117 |
| `corpus/privacy_qa` | Privacy policies | 3 |
| `benchmarks/*.json` | Questions, ground-truth answers and source spans (`contractnli`, `cuad`, `maud`) | 6,695 questions |

## Requirements

- Python 3.10+ and Jupyter
- An OpenAI API key, set as `OPENAI_API_KEY` (if missing, the notebook prompts for it; never paste it into the notebook)
- Packages (installed in the notebook's first cells): `langchain-openai`, `langchain-community`, `langchain-chroma`, `langchain-text-splitters`, `datasets`, `ragas`, `rouge_score`, `nltk`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

## How to run

1. Extract `rag_legal.zip` next to the notebook.
2. Edit the path constants in the notebook to match your machine: `CORPUS_DIR`, `PERSIST_DIR` (Chroma output) and `BENCHMARK_DIR`.
3. Run the notebook top to bottom.
   - Embedding all 105,267 chunks calls the OpenAI API and takes a while and costs money. Set `MAX_CHUNKS` (e.g. `2000`) in the indexing cell to test on a subset first.
   - Indexing is resumable: chunk ids are stable, and chunks already in Chroma are skipped.
4. Use `answer_question("...")` to ask a question; it prints the answer and the source files and character spans.

## Notebook outline

1. **Data loading, preparation and analysis**: load the `.txt` files, clean the text, compute document-length and word-frequency statistics, and compare documents with TF-IDF cosine similarity.
2. **Vector database and RAG chain**: chunk, embed, store in Chroma, build the retrieve-and-generate chain.
3. **Evaluation**: Ragas metrics plus retrieval hit rate and MRR on 10 random benchmark questions.
4. **Conclusions and insights**.

## Key findings

- The corpus clusters by document type: NDAs are about 0.45 similar to each other under TF-IDF, and two merger agreements reached 0.87. `maud` documents are long, so they produce 69,730 of the 105,267 chunks.
- On 10 sampled questions (seed 42): faithfulness 0.48, answer relevancy 0.33, context precision 0.52, context recall 0.56, hit rate@5 0.10, MRR 0.05.
- Retrieval is the weak point. Questions name a specific contract, but search runs over all 644 documents and often returns similar clauses from other contracts. The model usually refuses rather than guesses, which is the intended behaviour.
- The sample is small and some Ragas calls timed out, so the scores show a pattern rather than a precise measurement.

Recommended improvements: filter retrieval by the contract named in the question, prefix chunks with the document title, add hybrid (BM25 plus vector) search and a re-ranker, use section-aware chunking for `maud`, retrieve more chunks before re-ranking, and re-evaluate on a larger stratified sample.
