Large-Scale & Resumable Embedding
=================================

``model.encode`` returns all embeddings at once, so a long job holds the whole corpus and the whole output
matrix in memory, and a crash loses all of it. :class:`~sentence_transformers.util.EmbeddingStore` writes
embeddings to a directory of shard files plus a manifest instead: completed shards survive a crash, a restart
only encodes what is missing, and the result is row-for-row identical to a single uninterrupted run.

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sentence_transformers.util import EmbeddingStore

   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

   # One shard per 50,000 rows. The store directory is safe to point a restarted job at.
   store = EmbeddingStore("product-embeddings", shard_size=50_000)
   stats = store.run(model, "products.txt", normalize_embeddings=True, batch_size=64)
   print(stats.num_rows, stats.num_shards_written, stats.num_duplicates_dropped)

   # Load the full result back, in the original row order.
   ids, embeddings = store.load()

If the process is killed three quarters of the way through, call ``run`` again with the same arguments:

.. code-block:: python

   stats = store.run(model, "products.txt", normalize_embeddings=True, batch_size=64)
   print(stats.num_shards_written, stats.num_shards_reused)  # only the missing shards are encoded

Corpus forms
------------

``run`` accepts four corpus shapes:

* a list of texts, identified by ``"0"``, ``"1"``, ...;
* a list of ``(identifier, text)`` pairs;
* a mapping of identifier to text;
* a path to a local UTF-8 text file, one text per line, identified by ``"line:0"``, ``"line:1"``, ...

Text files are read with undecodable bytes replaced, so dirty exports do not abort the job. Repeated
identifiers are stored once, at their first position. Use ``text_normalization="nfc-strip-control"`` to
normalize Unicode and drop control characters before encoding.

Inspecting and verifying a store
--------------------------------

.. code-block:: python

   report = store.verify(model=model)
   print(report.ok)
   print(report)  # rows, shards, dimension, fingerprint, and every difference found

``verify`` never re-encodes. It checks that every shard is present, that its file holds the recorded number of
rows and matches its SHA-256, that shard row ranges tile the store without gaps or overlaps, that no identifier
appears twice, and that the fingerprint matches the model and settings you pass in. A store whose shard was
truncated by a full disk is reported as incomplete and is re-encoded by the next ``run``.

Incremental updates
-------------------

Add, modify and delete rows without recomputing the rest:

.. code-block:: python

   stats = store.apply_delta(
       model,
       upserts=[("sku/1001", "New title and description"), ("sku/2042", "Updated description")],
       deletes=["sku/777"],
   )
   print(stats.num_added, stats.num_modified, stats.num_deleted)

Modified rows keep their position, deleted rows disappear, and new rows are appended in input order, so the
store matches what a full re-encode of the updated corpus would have produced. Untouched rows are copied from
the existing shards rather than re-encoded; shards with no change are reused untouched.

Fingerprints
------------

Each store records an :class:`~sentence_transformers.util.EmbeddingSpec` fingerprint over the model name,
revision and a digest of the model weights, the resolved prompt/template, normalization, dimension truncation,
output precision and text normalization. Resuming or extending with anything different raises
:class:`~sentence_transformers.util.FingerprintMismatchError` and names the differing fields; it never silently
mixes vectors from two configurations. A corpus whose identifiers or texts changed is likewise refused with
:class:`~sentence_transformers.util.PlanMismatchError`, pointing at ``apply_delta`` where appropriate.

Parallel producers
------------------

Shard ownership is fixed by row range, so several workers or machines can encode disjoint shards of one store:

.. code-block:: python

   store.run(model, corpus, shard_ids=range(worker_index, num_shards, num_workers))

Each shard is claimed with a kernel file lock while it is written and only recorded in the manifest once its
file is complete. A second worker that finds a shard claimed by a live process raises
:class:`~sentence_transformers.util.ShardClaimedError` instead of writing a competing copy. Because the manifest
is replaced atomically, concurrent producers can never produce duplicate rows or a half-written shard that
counts as done.

Recipe: finishing a killed 420k-row job
---------------------------------------

.. code-block:: python

   store = EmbeddingStore("catalog-embeddings", shard_size=20_000)

   # First attempt (was killed after ~7 hours).
   store.run(model, corpus, batch_size=64, normalize_embeddings=True)

   # After restart: encode only the missing shards.
   store.run(model, corpus, batch_size=64, normalize_embeddings=True)

   # Validate without re-encoding, then hand the vectors to the search index.
   assert store.verify(model=model).ok
   ids, embeddings = store.load()
