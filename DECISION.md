# DECISION.md: RAG pipeline change log

Purpose: record every change to the pipeline in `RAG_Assg_Legal_Documents_Starter.ipynb` and the evaluation results that go with it, so that when `context_recall` or `context_precision` moves, we know which change caused it.

**How to use this file**

* Add one entry per change under "Change log", newest at the bottom. Change one thing at a time, and re-run the evaluation after each change.
* Fill in the results table with the same questions and the same seed, otherwise scores are not comparable.
* Do not delete old entries. If a change is reverted, add a new entry saying so.

***

## 1\. Pipeline baseline \(state before any recorded change\)

| Stage | Setting |
| ----- | ------- |
| Loading | `DirectoryLoader` + `TextLoader` (`autodetect_encoding`), paths from `.env` (`CORPUS_DIR`, `PERSIST_DIR`, `BENCHMARK_DIR`) |
| Cleaning | Emails/URLs/phones removed, lowercased, stopwords removed. **Used for EDA only**; chunking uses the original text |
| Chunking | `RecursiveCharacterTextSplitter`, `chunk_size=2000`, `chunk_overlap=500`, separators `["\n\n", "\n", ". ", " ", ""]`, `add_start_index=True` |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector store | Chroma, collection `legal_docs`, chunk id = `rel_path::start_index` |
| Retriever | Find the contract named before `;` in the query (TF-IDF over file name + first 2000 chars), then dense `similarity_search` **k=5** restricted to that file. Unfiltered if the query has no `;` |
| LLM | `gpt-4o-mini`, temperature 0, answer only from context |
| Evaluation | Ragas `ContextRecall` (LLM `gpt-4o-mini`), retrieval only, 10 questions sampled with `random.seed(42)` from the answerable benchmark questions. `ContextPrecision` is written but commented out |

Baseline results are in `ragas_eval/experiments/retrieval_top5.csv`.

***

## 2\. Results

Keep this table up to date. Same questions and seed for every row.

| # | Change | Retriever | Questions | context\_recall (mean) | context\_precision (mean) | Result file | Notes |
| --- | ------ | --------- | --------- | --------------------- | ------------------------ | ----------- | ----- |
| 1 | Baseline without contract filter | Dense, k=5 | 50 queries drawn at random from the benchmark files | **0.6866** | **0.62938** | `ragas_eval/experiments/dense_only_*.csv` | Numbers reported by the user. Plain dense search over the whole DB, no filter by the contract named in the query. Row 0 adds that filter |
| 2 | Baseline | Dense, filtered by the contract named in the query, k=5 | 50 queries drawn at random from the benchmark files | **0.92333** | **0.6760** | `ragas_eval/experiments/retrieval_top5.csv` | Numbers reported by the user. Not comparable with rows 1-2 unless the query set is the same: confirm which queries were used |
| 3 | Hybrid BM25 + Cohere rerank (later run, config to confirm) | Dense + BM25 -> RRF -> Cohere rerank | 50 queries drawn at random from the benchmark files | **0.95** | **0.791888** | `ragas_eval/experiments/hybrid_rerank_b*.csv` | Numbers reported by the user as "mean context recall / precision after BM25 and Cohere reranking". Which DB and settings produced this is not recorded: fill in before using it to credit C3 or C4 |
| 4 | Unique chunk ids + DB rebuild (see C3) | Same as row 1 | same 50 queries (use the same sample as row 1) | *not run yet* | *not run yet* | *fill in* | Compare against row 1 to see how much was lost to dropped duplicate-id chunks |
| 5 | Contextual headers + clause-aware parents + parent-child retrieval + overlap de-duplication (see C6, C7, C8) | Dense + BM25 over 500-char child chunks (with contract header) -> RRF -> Cohere rerank -> mapped to parent clauses -> overlap de-dup, final k=5 parents | 50 queries drawn at random from the benchmark files | **0.945** | **0.8337499** | *fill in* | Numbers reported by the user. Summary of changes: contracts split on clause headings into parents (300-3000 chars); parents split into 500/100 children that carry a `Contract / Type / Title` header and section heading; children embedded in a new Chroma folder (`<PERSIST_DIR>_pc`); the LLM and Ragas receive the parent clauses. Versus row 1: recall 0.95 -> 0.945 (no real change), precision 0.791888 -> 0.8337499 (+0.042). Contexts are clause-sized now, so precision is not strictly comparable. These three changes were evaluated together, so the effect of each one is not separated |
| 6 | Same pipeline as row 4, full corpus, **benchmark run without random sampling** | Same as row 3: Dense + BM25 over 500-char child chunks (with contract header) -> RRF -> Cohere rerank -> mapped to parent clauses -> overlap de-dup, final k=5 parents | 50 queries drawn at random from the benchmark files | **0.938333** | **0.8552** | *fill in* | Numbers reported by the user. Versus row 4 (random 50): recall 0.93 -> 0.938 (+0.008), precision 0.708 -> 0.855 (+0.147). Versus row 3 (small corpus, first 50 contractnli): recall 0.945 -> 0.938 (-0.007, within noise), precision 0.834 -> 0.855 (+0.021). Adding data to the vector database did not reduce quality with this retriever; the low precision in row 4 came from that particular random sample of 50 queries |
| 7 | Full corpus + full benchmark, 50 random queries (mixed benchmarks) | Same as row 3: Dense + BM25 over 500-char child chunks (with contract header) -> RRF -> Cohere rerank -> mapped to parent clauses -> overlap de-dup, final k=5 parents | 50 queries drawn at random from the full benchmark files, so they cover different documents in the full corpus | **0.93** | **0.7080** | *fill in* | Numbers reported by the user. First run on the full corpus and not only contractnli, so it is closer to real use. Not comparable with rows 0-3b (different corpus and queries). The user confirmed the retriever strategy was the same as row 3 (parent-child pipeline). Only the corpus (more data in the vector database) and the queries changed. Row 5 re-runs the benchmark without random sampling; it shows the drop in this row did not persist\* |
| 8 | Same pipeline as row 3, evaluated with the **soft contract name, no filter** (see E1) | Same as row 3, but `find_contract` returns `None`, so the search covers the whole corpus | Same 50 cases as row 3, `;` removed from the query | **0.915** | **0.8259** | `ragas_eval/experiments/softname_parent_child_b*.csv` | Numbers reported by the user (row 3 re-reported as 0.945 / 0.8337 in the same run). Without the file filter recall drops by 0.03 and precision by about 0.008. The contract description is still in the query text, so this is not a fully name-free test, and the 50 cases are all contractnli |

***

## 3\. Change log

### C1. Hybrid retrieval: BM25 + Cohere reranking

* **Status:** applied and evaluated once (row 1: recall 0.95, precision 0.791888). Against the dense-only baseline (row 0: recall 0.92333, precision 0.6760) recall rose slightly and precision rose clearly (0.6760 to 0.791888), but the two runs may not use the same queries, so the gain is not confirmed.
* **Why:** dense-only top-5 can miss clauses where exact wording matters (names, defined terms, numbers). The idea was already noted in the old cell 73.
* **What changed (notebook):**
    * Old retriever code in the RAG chain cell (`retrieve`, `retriever = RunnableLambda(retrieve)`, the `rag_chain` assembly and its quick test) is **commented out, not deleted**. `find_contract`, the TF-IDF header index, LLM, prompt, `format_docs` and `answer_chain` are unchanged.
    * New section **2.2.1b**, placed right after the RAG chain cell, adds:
        * `pip install rank_bm25 langchain-cohere` and `COHERE_API_KEY` loading (env, `getpass` fallback).
        * Per-contract BM25 indexes (`BM25Okapi`, lowercase alphanumeric tokens), built lazily and cached. Whole-corpus BM25 is the fallback when the query names no contract.
        * `hybrid_retrieve(query, k=5, fetch_k=20)`:
            1. `find_contract` (unchanged).
            2. Dense search (full query) and BM25 search (question part after `;`), each top 20 inside the matched contract.
            3. Reciprocal Rank Fusion with constant 60, deduplicated by chunk id, keeping up to 40 candidates.
            4. Cohere `rerank-english-v3.0` on the question part, keep top `k=5`. If the Cohere call fails, fall back to the fused order and log a warning.
        * `retriever` and `rag_chain` are rebuilt with the new retriever.
    * The old "Steps for improving Recall" note (cell 73) now points to section 2.2.1b.
    * A recall comparison cell (dense vs hybrid) was added after the section.
* **Parameters worth tuning later:** `fetch_k` (20), RRF constant (60), candidates passed to the reranker (40), rerank model, final `k` (5), whether BM25/rerank use the question part only.
* **Expected effect:** recall should rise because BM25 recovers exact-term matches the embeddings miss, and the reranker moves the best chunk into the top 5. Precision may also rise since the reranker orders by relevance.
* **Measured effect:** *fill in after the run (row 1 in the table).*

### C2. Batched Ragas evaluation (10 rows at a time)

* **Status:** **applied by the user, by hand** in the section 3.1.3 evaluation cell (my own edit was interrupted, so the code differs from the plan below). As found in the notebook: `BATCH_SIZE = 5` (not 10), `sample = all_rows if len(all_rows) <= 50 else random.sample(all_rows, 50)`, a loop that creates one `Dataset` per batch and awaits `run_experiment.arun(batch_ds, name=f"hybrid_rerank_b{batch_no}")`. The sample is **not seeded** in that cell unless `random.seed` is set earlier, so check that the same 50 queries are used for every comparison.
* **Why:** evaluate more than 10 questions without launching them all at once, and limit API rate-limit and cost spikes.
* **Planned design:**
    * `run_in_batches(rows, experiment_name, batch_size=10)`: for each slice of 10 rows, build a `Dataset`, `await run_experiment.arun(...)`, save as `<name>_b1`, `_b2`, ... and skip batches whose CSV already exists (resume support). Results are concatenated.
    * Sampling line requested by the user: `sample = all_rows if len(all_rows) <= 50 else random.sample(all_rows, 50)`, where `all_rows` is the list of answerable benchmark questions.
    * The old single-run code is to be commented out and kept.
* **Effect on scores:** none by itself. It only changes how the evaluation is run. **Caveat:** moving from 10 questions (seed 42) to up to 50 random questions changes the evaluation set. A recall difference between runs on different question sets is **not** evidence that a retriever change worked. Re-run the baseline on the same 50 questions before comparing (add a row to the table).

### C3. Unique chunk ids with `chunk_number` (section 2.1.2)

* **Status:** code applied by the user in the notebook; **DB not yet rebuilt and not yet evaluated.**
* **Why:** chunk ids were `rel_path::start_index`. LangChain finds `start_index` by searching the original text for each chunk, so repeated passages (or a failed search giving `-1`) can give two chunks of one file the same `start_index`. The insert cell silently skipped repeated ids, so those chunks were **never embedded**. Their text could not be retrieved, which is a suspected cause of the remaining recall loss (observed on one file: the file was matched correctly but the relevant chunks were absent).
* **What changed (notebook):** in the cell that builds `chunk_ids`, a per-file `Counter` now gives each chunk a running `chunk_number` (0, 1, 2, ... within its file), stored in `chunk.metadata['chunk_number']`. New id: `rel_path::chunk_number::start_index`. An earlier attempt counted in a separate loop first, so every chunk of a file got the same number; this was corrected by counting and building the id in one pass.
* **Needs a DB rebuild:** the old DB used the old ids. It was renamed to `chroma_db_small_old` (6.6 GB) as a backup. The new DB must be built from scratch with the new ids, which re-embeds every chunk (OpenAI cost). Chunking itself is unchanged (2000/500).
* **Expected effect:** recall should rise, in proportion to how many chunks had been dropped. Count them before the rebuild if possible: chunks with `start_index == -1` and repeated old ids.
* **Measured effect:** *fill in (row 2 in the table).*
* **Loose end:** `chunk_id_of` in the hybrid retriever section (cell `hyb_bm25`) still builds its RRF de-duplication key as `rel_path::start_index`. Two distinct chunks that share a `start_index` would be merged there and one lost. Switch it to include `chunk_number` (as in C3) so the retriever matches the DB ids.

### C4. Smaller chunks (1000 / 200)

* **Status:** planned (not applied, DB not rebuilt).
* **Why:** a clause can be diluted inside a 2000-character chunk, which hurts both the embedding and the BM25 score. Smaller chunks should rank better on specific clauses.
* **What changes:** `chunk_size=1000, chunk_overlap=200` in the chunking cell (1.4.1). Everything after it must be rebuilt: new Chroma folder (set `PERSIST_DIR` to a new name, e.g. `chroma_db_small_1000`, so the 2000/500 DB stays comparable), re-embed, and re-run the BM25 cell (`chunks_by_path` and `_bm25_cache` are built from `chunks`).
* **Cost note:** roughly 1.9x as many chunks, but about the same total embedded text, so OpenAI cost is similar and the insert takes longer.
* **Expected effect:** higher recall for clause-level questions. Context per retrieved chunk halves, so the LLM sees less text unless `k` or C5 compensates.
* **Evaluate separately from C5**, using the same 50 queries as rows 1-2 in the table. Also reset `fetch_k` and `k` to the same values used in the run you compare against (the user raised them to 50 and 30 during debugging, which changes results on its own).
* **Measured effect:** *fill in.*

### C5. Neighbour-chunk expansion

* **Status:** applied, evaluated, **reverted**. Recall did not improve much, so the retriever is back to plain `hybrid_retrieve`. The code is kept in the notebook, commented out. Exact before/after numbers were not recorded here: *fill in if you have them*.
* **Why:** an answer split across a chunk boundary is only partly retrieved. Adding the chunks before and after each hit restores it.
* **What changes:** after reranking, for each final chunk also include the previous and next chunk of the same file (`chunk_number` +/- 1, taken from `chunks_by_path`), de-duplicated, hits first. Wrap as a new retriever and rebind `retriever` / `rag_chain`. Needs `chunk_number` (C3).
* **Expected effect:** higher recall, lower precision (extra chunks are mostly not relevant) and up to 3x more context sent to the LLM.
* **Measured effect:** *fill in (compare with and without C5 on the same queries).*

### C6. Contextual headers + clause-aware chunking (notebook section 2.3)

* **Status:** code added; **DB not built and not evaluated yet.**
* **Why:** fixed 2000-character chunks cut clauses in half, dilute a short clause inside unrelated text, and say nothing about which contract they belong to (which hurts questions that do not name the contract).
* **What changed:** the text of each contract is split on clause headings (`Article 5`, `Section 2.1`, `12.3 ...`). Clauses under 300 characters are merged into the next, clauses over 3000 are split without overlap. Each contract gets a header `Contract: <file> | Type: <folder> | Title: <first line>` which is prepended to every searchable chunk (section 2.4). Offsets are exact (no `-1`), and ids are unique by construction (`<file>::p<n>::c<m>`), so the duplicate-id problem from C3 cannot recur here.
* **Superseded and commented out (kept for reference):** old chunking cell (2000/500), old Chroma creation / count / duplicate-id / insert cells, old BM25 cell and old hybrid retriever + `rag_chain` cell. C3 (unique ids by `chunk_number`) and C4 (planned 1000/200) therefore apply to the old pipeline only.
* **Offline check** (all 126 files of `corpus_small`, no API calls): 10,135 parents (average 1,449 chars), parent offsets 0 wrong, ids unique, parents cover 99.7% of the text. 5 of 48,595 children have an approximate start offset (a repeated passage); this only matters for overlap de-duplication.
* **Expected effect:** higher recall and precision for clause-level questions; better results when the question does not name the contract.
* **Measured effect:** *fill in.*

### C7. Parent-child retrieval: small chunks for search, big chunks for answers (section 2.4)

* **Status:** code added; **DB not built and not evaluated yet.**
* **What changed:** each parent is split into 500-char children (100 overlap), 48,595 in total on `corpus_small`. Children (with header + section heading) are embedded into a **new** Chroma folder `<PERSIST_DIR>_pc` (collection `legal_docs_pc`); the old DB is untouched. Search is dense + BM25 inside the contract found by `find_contract` (unchanged), Reciprocal Rank Fusion, then Cohere rerank of the children (`fetch_k=30`, `child_k=10`). BM25 also indexes the header text.
* **Cost note:** roughly 6.5M tokens to embed with `text-embedding-3-small`, which should be small (about 0.13 USD at 0.02 USD per million tokens; an estimate, check current pricing).
* **Parameters to tune later:** child size / overlap (500/100), parent min/max (300/3000), `fetch_k`, `child_k`, RRF constant.
* **Measured effect:** *fill in.*

### C8. Overlap de-duplication and final retriever (section 2.5)

* **Status:** code added; **not evaluated yet.**
* **What changed:** reranked children are mapped to their parent clauses, then `remove_overlap_duplicates` drops later passages of the same file that overlap a kept one by 50% or more of the shorter span. Final `k=5` distinct parents go to the LLM and to Ragas. `retriever` and `rag_chain` are rebuilt on this.
* **Expected effect:** fewer repeated contexts, higher precision. Contexts are now clause-sized (up to 3000 chars), so Ragas precision may behave differently than with the old 2000-char chunks: compare with care.
* **Measured effect:** *fill in (use a new experiment name, e.g. `parent_child_b{n}`; same 50 queries, seeded).*

### E1. Evaluation without a hard contract filter (notebook section 3.2)

* **Status:** Ragas part run for the soft-name set (row 3b). The file-level hit rates and the stripped set are **not recorded yet**. This is an evaluation set, not a pipeline change.
* **What it does:** builds two question sets from the same sample as the main evaluation. *Soft name:* the contract description stays in the text but the `;` is removed, so `find_contract` does not filter and the search covers the whole corpus. *Stripped:* only the text after `;`. Reports the file-level hit rate (`hit_top1`, `hit_any`, share of results from a ground-truth file) for both, and Ragas recall / precision for the soft set.
* **Finding while building it:** the benchmark questions are templated. In `benchmarks_small` there are only **17 distinct question texts for 977 contractnli cases, 35 for 4,042 cuad, 19 for 1,676 maud**, and in the first 50 cases 49 are identical to another case once the contract name is removed. So the *stripped* set is almost entirely ambiguous: its file hit rate only shows how much the contract name carries, not retriever quality. The *soft* set is the meaningful test.
* **Measured effect (soft name, no filter):** context recall 0.915 vs 0.945 with the filter (-0.03); context precision 0.8259 vs 0.8337 (-0.008). The drop is small, which suggests the hybrid retriever finds the right contract from the description kept in the query text. The 0.03 recall gap on 50 cases is within the range of Ragas noise, so confirm it on a larger, mixed-benchmark sample.
* **Measured effect (stripped, no name at all), file-level hit rate on the same 50 cases:** `hit_top1` 0.020, `hit_any` 0.040, `share_from_truth` 0.008 (all contractnli). With 126 files in the corpus, picking 5 files at random would hit a given file about 4% of the time, so the retriever is at chance level here. This is expected, not a defect: 49 of the 50 stripped questions are identical to another case, so the question does not identify the contract. It confirms that the contract name (even loosely, as in the soft set) is what lets the pipeline find the right file.
* **File hit rates for the soft-name set:** *fill in (printed in the same cell, above the stripped block).*

***

## 4\. Known issues / not yet done

* The main evaluation cell now uses `sample = all_rows[:50]`, the **first** 50 answerable cases, not a random sample. The benchmarks are loaded in the order contractnli, cuad, maud, so these 50 cases are **all contractnli** (checked on `benchmarks_small`). Row 0a-3 numbers therefore describe contractnli only. Use a seeded random sample across the three benchmarks (or per-benchmark samples) before drawing general conclusions.
* Cell 30 has `random_10 = ...` commented out, so cell 31 fails on a fresh run (pre-existing).
* `ContextPrecision` is commented out, so precision has never been measured. Re-enable it (it needs a call per retrieved chunk, so it is costlier) before claiming any precision change.
* Only retrieval is evaluated. Answer quality (faithfulness, answer correctness) is not scored.
* The retriever relies on the benchmark's `Contract; question` format.
* The recall comparison cell (`hyb_eval`) still re-runs the old single-dataset experiment (`run_experiment.arun(dataset, ...)`), so it does not use the batched results from C2. Update it before relying on it.
* Further ideas from the recall analysis, not tried yet (log each as its own change): raise the final `k` to 8-10, raise `fetch_k` to 40-50, better contract matching in `find_contract`, neighbour-chunk expansion, smaller or clause-aware chunks, query expansion, `text-embedding-3-large`, a newer Cohere reranker, weighted RRF.
* A backup of the notebook from before C1 is at `C:\Users\rituP\AppData\Local\Temp\claude\01555b876cee0a91\d5220a42-1b76-4c13-aaf9-ca58088c6312\scratchpad\nb_backup.ipynb`. It is in a temp folder, so move it somewhere permanent if you want to keep it.

***

## 5\. Template for new entries

```
### Cn. <short title>
- Status: planned | applied | evaluated | reverted
- Why:
- What changed (cells / files):
- Parameters:
- Expected effect:
- Measured effect: context_recall X -> Y, context_precision X -> Y (questions, seed, res
```