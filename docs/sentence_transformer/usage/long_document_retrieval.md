Long Document Retrieval
=======================

:func:`~sentence_transformers.util.retrieval.semantic_search` compares a query against a
single embedding per corpus entry. When the corpus contains long documents, compressing each
document into one embedding loses the location of the relevant passage, and encoding every
chunk as its own corpus entry makes several chunks of the same document compete with each
other for the top of the ranking.

:class:`~sentence_transformers.util.document_retrieval.DocumentRetriever` adds a
document-aware layer on top of the unchanged ``encode`` and ``semantic_search`` code:

- documents are split on sentence boundaries, with overlap between neighbouring chunks
  (:mod:`~sentence_transformers.util.chunking`),
- every chunk is encoded through ``model.encode`` exactly as before,
- chunk scores are aggregated into one document score with a configurable strategy,
- each document gets at most one result, with the matching chunks and snippets attached.

Basic usage
-----------

.. code-block:: python

   from sentence_transformers import DocumentRetriever, ChunkingConfig

   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   retriever = DocumentRetriever(
       model,
       ChunkingConfig(max_chars=1000, overlap_chars=100),
       aggregation="mean_top_n",
       mean_top_n=3,
   )
   retriever.add_documents({"manual": long_manual_text, "faq": faq_text})

   for hit in retriever.search("What is the default of parameter A?", top_k=5)[0]:
       print(hit.doc_id, hit.score, hit.coverage)
       chunk = hit.best_chunk
       print(f"chars [{chunk.start_char}, {chunk.end_char})", hit.snippets[0])

Every :class:`~sentence_transformers.util.document_retrieval.DocumentHit` carries the
character range of the chunk that produced the score, so the snippet can always be traced
back to the source document.

Aggregation strategies
----------------------

``aggregation`` controls how chunk scores become one document score:

- ``"max"``: the best chunk. Fast, but a long document with one lucky chunk can outrank a
  document that is relevant throughout.
- ``"mean_top_n"``: the mean of the top ``mean_top_n`` chunks. Rewards documents whose
  relevance is spread over several passages and damps single-chunk luck.
- ``"hit_count_weighted"``: moves the best chunk score towards the top of the similarity
  range in proportion to the fraction of chunks above ``hit_threshold``. Bounded, and it
  never inverts on negative similarities.

Chunk parameters and short documents
------------------------------------

Chunks never end in the middle of a sentence. A sentence longer than ``max_chars`` is split
at clause punctuation and then at whitespace, and only as a last resort at a hard character
cut (reported through ``ChunkPlan.forced_chunks``). A document that fits into one chunk is
not split at all, so behaviour for short documents is unchanged.

Setting ``max_chunks_per_doc`` caps the index size per document; when it triggers, the plan
records the dropped character range and a warning names the document and the impact.

Persistence and incremental updates
-----------------------------------

.. code-block:: python

   retriever.save("index/")
   retriever = DocumentRetriever.load("index/", model)

   # Only documents whose text changed are re-chunked and re-encoded.
   retriever.add_documents({"manual": updated_manual_text})

Saving and loading restores the chunk plans and chunk embeddings verbatim, so queries never
re-chunk or re-encode the corpus. Encoding failures are transactional: a document that fails
keeps its previous chunks and vectors, and the error names the document, the chunk index and
the character offset.

Disabling the feature
---------------------

Pass ``enabled=False`` to fall back to one whole-document embedding per document and plain
:func:`~sentence_transformers.util.retrieval.semantic_search` scoring. The resulting hits,
scores and order match the pre-chunking behaviour exactly.
