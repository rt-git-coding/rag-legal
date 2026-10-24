# Architecture

This describes the RAG system built in `RAG_Legal_Docs.ipynb`. It is a single-notebook, batch-indexed pipeline with a persistent local vector store.

## Overview

```
 corpus/*.txt                                         benchmarks/*.json
      |                                                      |
      v                                                      v
 DirectoryLoader + TextLoader  (644 Documents + metadata)   questions / ground truths / spans
      |                       \                              |
      |                        +--> clean_text -> EDA         |
      |                            (lengths, word freq,      |
      v                             TF-IDF similarity)       |
 RecursiveCharacterTextSplitter (1000 chars, 200 overlap)    |
      |  original text, start_index kept                     |
      v                                                      |
 OpenAIEmbeddings (text-embedding-3-small, 1536-d)           |
      v                                                      |
 Chroma  collection "legal_docs" (persisted on disk)         |
      ^                                                      |
      | top-5 similarity search                              |
 question --> retriever --> format_docs --> prompt --> gpt-4o-mini --> answer + sources
                                                              |
                                                              v
                                       Evaluation: Ragas + hit rate@5 + MRR
```

## Components

### 1. Ingestion
- `load_files_as_documents` uses LangChain `DirectoryLoader` with `TextLoader` (`autodetect_encoding=True`, `silent_errors=True`) over `**/*.txt`.
- Each document gets metadata: `source`, `file_name`, `folder` (document type) and `rel_path`. `rel_path` matches the `file_path` format in the benchmark files.

### 2. Preprocessing (analysis only)
- `clean_text` removes emails, URLs and phone numbers, lowercases, strips special characters, collapses whitespace and drops NLTK stop words.
- `preprocess_documents` logs and skips empty or corrupted files (none were skipped) and keeps the raw text in `metadata["original_text"]`.
- The cleaned text feeds only the exploratory analysis (length statistics, word frequencies, TF-IDF). It is not what gets embedded.

### 3. Chunking
- `RecursiveCharacterTextSplitter`: 1,000 characters, 200 overlap, separators `\n\n`, `\n`, `. `, space, empty string.
- It splits the original, uncleaned documents. Embedding and generation work better on natural text, and the benchmark answer spans are character offsets into the original text.
- `add_start_index=True` stores each chunk's offset in `metadata["start_index"]`.
- Result: 105,267 chunks, 648 characters on average (`maud` 69,730; `cuad` 34,018; `contractnli` 1,444; `privacy_qa` 75).

### 4. Vector store
- Embeddings: OpenAI `text-embedding-3-small` (1536 dimensions).
- Store: Chroma collection `legal_docs`, persisted to `PERSIST_DIR`.
- Chunk id is `rel_path::start_index`, which is stable and unique. Re-running does not create duplicates.
- Indexing runs in batches of 500, skips ids already stored, and on a duplicate-id error retries record by record, so an interrupted run can resume.

### 5. RAG chain
- Retriever: Chroma similarity search, `k=5`.
- LLM: `gpt-4o-mini`, temperature 0.
- Prompt: the system message restricts answers to the supplied context and tells the model to say so when the answer isn't there. Each chunk is labelled `[Source: file_name]`.
- The chain is built from LangChain runnables: `RunnableParallel(source_documents=retriever, question=passthrough)`, then `.assign(context=...)`, then `.assign(answer=prompt | llm | StrOutputParser())`. Its output holds the question, the retrieved chunks, the context and the answer.
- `answer_question(question)` wraps the chain. It returns the answer, the retrieved documents, the context texts, and each source's `file_path` and character `span` in benchmark format.

### 6. Evaluation
- The benchmark loader flattens the three JSON files into 6,695 items (`cuad` 4,042, `maud` 1,676, `contractnli` 977). Each item has the question, the ground-truth snippets (file, span, answer) and a joined reference answer.
- `evaluate_rag(indices)` runs the pipeline on the chosen questions and scores them with:
  - **Ragas** (judge: `gpt-4o-mini`, embeddings: the same OpenAI model): Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall.
  - **Retrieval metrics**: hit rate@5 and MRR. A retrieved chunk counts as relevant if it comes from a ground-truth file and its character span overlaps a ground-truth span.
- The notebook runs it on 10 questions sampled with seed 42, because a full run is costly.

## Design decisions and trade-offs

| Decision | Reason | Trade-off |
|---|---|---|
| Embed original text, not the cleaned text | Natural text embeds and generates better; spans stay valid | Cleaned text is used only for EDA |
| 1,000-character chunks with overlap | Keeps clauses mostly intact within a modest embedding cost | Long `maud` clauses are cut across chunks |
| One collection for all four document types | Simple, and matches the benchmark | `maud` dominates the index, and similar boilerplate competes across contracts |
| Stable chunk ids | Idempotent, resumable indexing | Chunks with an identical path and offset are treated as duplicates |
| Answer only from context | Reduces hallucination | The model refuses when retrieval misses |

## Known limitations

- Retrieval is the bottleneck: hit rate@5 was 0.10 on the sample. Questions name a specific contract, but the search spans all documents and returns similar clauses from others.
- No metadata filter, hybrid search or re-ranker yet.
- The evaluation sample is only 10 questions, and some Ragas calls timed out, leaving context precision empty for 5 questions.
- Hit rate is strict. A different chunk from the right file that answers the question still counts as a miss.
- Local paths are hard-coded as constants in the notebook (`CORPUS_DIR`, `PERSIST_DIR`, `BENCHMARK_DIR`) and must be edited per machine.

## Suggested extensions

1. Extract the contract or party name from the question and filter retrieval by `file_name`/`rel_path` metadata.
2. Prefix each chunk with its document title before embedding.
3. Combine BM25 keyword search with vector search, then re-rank, retrieving more than 5 candidates first.
4. Use section-aware chunking for merger agreements.
5. Evaluate on a larger sample stratified by benchmark, with lower Ragas concurrency or a higher timeout.
