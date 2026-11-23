# reports/

Acceptance evidence for the long-document chunked retrieval feature
(`sentence_transformers.util.chunking` and `sentence_transformers.util.document_retrieval`).

## Reproduce

```bash
# from the repository root, with the cached stsb-bert-tiny model available
python reports/run_acceptance.py

# the unit/acceptance tests that back the report
python -m pytest tests/util/test_chunking.py tests/util/test_document_retrieval.py -v
python -m pytest tests/util/test_retrieval.py -v
```

The script downloads nothing. It seeds `torch.manual_seed(0)` and uses the
`stsb-bert-tiny-safetensors` model that the test suite already relies on, so repeated
runs produce the same chunk boundaries and the same rankings.

## Files

| File | Covers |
| --- | --- |
| `chunk_boundaries.json` | Chunk spans and a per-cut check that no cut lands inside a sentence (CJK and ASCII documents). |
| `dedup_before_after.json` | Top-10 ranking before (chunk level) and after (document level) deduplication. |
| `aggregation_comparison.json` | Ranking under `max`, `mean_top_n` and `hit_count_weighted`. |
| `incremental_update.json` | Bit-for-bit comparison of every untouched document's vectors after updating one document. |
| `disabled_mode_parity.json` | The disabled retriever vs. `semantic_search` over whole-document embeddings. |
| `benchmark.json` | Indexing and query timings at 1,000 documents. |
| `pytest_existing_encode_and_retrieval.txt` | Actual output of `pytest tests/sentence_transformer/test_compute_embeddings.py tests/util/test_retrieval.py tests/util/test_similarity.py tests/util/test_chunking.py tests/util/test_document_retrieval.py -v`. |
| `acceptance_report.md` | Human-readable summary of the above. |

## How the tests map to the requirements

| Requirement | Test |
| --- | --- |
| 1. Sentence-boundary chunks with overlap | `test_cut_points_never_land_inside_a_sentence`, `test_cut_points_never_land_inside_an_english_sentence`, `test_overlap_preserves_context_across_boundaries` |
| 2. Source locations per chunk | `test_chunks_carry_source_locations`, `test_hits_cite_the_chunks_that_produced_the_score` |
| 3. Configurable document-level aggregation | `test_aggregation_changes_the_order_as_expected`, `test_mean_top_n_matches_manual_average`, `test_max_aggregation_uses_the_best_chunk_only` |
| 4. One slot per document | `test_dedup_prevents_one_document_from_filling_the_results`, `test_results_are_deduplicated_per_document` |
| 5. Chunks + snippets in the result | `test_hits_cite_the_chunks_that_produced_the_score`, `test_snippet_length_is_controlled` |
| 6. Short documents unchanged | `test_short_document_is_not_split`, `test_short_documents_behave_like_before` |
| 7. Reproducible for the same input and parameters | `test_chunking_is_deterministic` |
| 8. Incremental update by document | `test_only_changed_documents_are_re_encoded`, `test_updating_one_document_leaves_other_vectors_bit_identical` |
| 9. Coverage metric | `test_coverage_measures_hit_fraction`, `test_covered_fraction_counts_overlaps_once` |
| 10. Disabled mode is unchanged | `test_disabled_mode_matches_whole_document_semantic_search`, `test_disabled_mode_does_not_change_encode` |
| 11. Persist and reuse without re-chunking | `test_save_and_load_round_trip`, `test_queries_do_not_re_chunk`, `test_loading_does_not_re_encode` |
| Failure names document and chunk, keeps old entry | `test_encoding_failure_names_document_and_chunk`, `test_failed_update_does_not_mix_old_and_new_entries` |
| Empty / oversized documents handled explicitly | `test_empty_document_is_reported_not_dropped`, `test_truncation_reports_the_dropped_range`, `test_truncation_is_visible_in_the_plan` |
| No mixed index after a mid-way failure | `test_failed_update_does_not_mix_old_and_new_entries`, `test_non_strict_mode_commits_healthy_documents` |
