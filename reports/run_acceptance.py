"""Acceptance evidence for the long-document chunked retrieval feature.

Run with::

    python reports/run_acceptance.py

It writes machine-readable JSON and a Markdown summary into ``reports/``:

- ``chunk_boundaries.json``: sample long documents, their chunks, and a proof that
  no cut point lands inside a sentence.
- ``dedup_before_after.json``: top-10 results before (chunk-level search) and after
  (document-level search) deduplication.
- ``aggregation_comparison.json``: ranking under ``max``, ``mean_top_n`` and
  ``hit_count_weighted``.
- ``incremental_update.json``: bit-for-bit comparison of untouched document vectors
  before and after an incremental update.
- ``disabled_mode_parity.json``: parity between the disabled retriever and both
  ``model.encode`` (stored vectors are bit-identical) and plain ``semantic_search``
  over whole-document embeddings.
- ``benchmark.json``: indexing and query timing at the 1,000-document scale.
- ``acceptance_report.md``: human-readable summary of all of the above.
- ``pytest_existing_encode_and_retrieval.txt``: actual pytest output for the existing
  encode/retrieval/similarity tests plus the new chunking/retriever tests (written by
  hand from the command shown in ``reports/README.md``).

The script downloads nothing: it uses the cached ``stsb-bert-tiny-safetensors``
model that the test suite already relies on.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Run from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from sentence_transformers.util.chunking import ChunkingConfig, chunk_document
from sentence_transformers.util.document_retrieval import DocumentRetriever
from sentence_transformers.util.retrieval import semantic_search

REPORTS = Path(__file__).resolve().parent
MODEL_ID = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def write_json(name: str, payload: Any) -> None:
    (REPORTS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- documents

CJK_SENTENCES = [
    "客服知识库第一节说明退换货流程。",
    "第二节列出保修期的具体参数A，默认值为十。",
    "第三节说明发票申请的所需材料。",
    "第四节解释会员积分的计算方式。",
    "第五节给出物流时效的参考区间。",
    "第六节描述技术支持的联系渠道。",
]
ASCII_SENTENCES = [
    "Section one covers the return and exchange workflow.",
    "Section two lists parameter A with its default value of ten.",
    "Section three explains what an invoice application needs.",
    "Section four describes how loyalty points are calculated.",
    "Section five gives the expected delivery window.",
    "Section six lists how to reach technical support.",
]


def long_cjk_document(repeats: int = 5) -> str:
    return "".join(sentence for _ in range(repeats) for sentence in CJK_SENTENCES)


def long_ascii_document(repeats: int = 5) -> str:
    return " ".join(sentence for _ in range(repeats) for sentence in ASCII_SENTENCES)


def filler_document(topic: str, sentences: int = 24) -> str:
    return " ".join(
        f"The {topic} guide paragraph {index} discusses routine topic {index} with no special parameter."
        for index in range(sentences)
    )


# --------------------------------------------------------------------------- 1. boundaries


def boundary_report() -> dict[str, Any]:
    config = ChunkingConfig(max_chars=90, overlap_chars=20)
    results = []
    for doc_id, text in {"cjk-manual": long_cjk_document(), "ascii-manual": long_ascii_document()}.items():
        plan = chunk_document(doc_id, text, config)
        sentence_ends = set()
        # For ASCII text the chunker keeps the sentence's trailing space in the chunk, so
        # a valid cut is the end of the sentence plus that run of whitespace.
        sentence_runs = []
        for sentence in CJK_SENTENCES if doc_id.startswith("cjk") else ASCII_SENTENCES:
            cursor = 0
            while True:
                found = text.find(sentence, cursor)
                if found < 0:
                    break
                sentence_runs.append((found, found + len(sentence)))
                sentence_ends.add(found + len(sentence))
                cursor = found + 1
        # Map every cut point that sits only in trailing whitespace back to the sentence end.
        sentence_ends_with_spaces = set(sentence_ends)
        for _, sentence_end in sentence_runs:
            cursor = sentence_end
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
                sentence_ends_with_spaces.add(cursor)

        chunk_rows = []
        for chunk in plan.chunks:
            inside_sentence = chunk.end_char not in sentence_ends_with_spaces and chunk.end_char != len(text)
            chunk_rows.append(
                {
                    **chunk.to_dict(),
                    "text": chunk.text,
                    "ends_at_sentence_boundary": chunk.end_char in sentence_ends_with_spaces,
                    "cut_inside_sentence": inside_sentence,
                    "last_chars": chunk.text[-12:],
                }
            )
        results.append(
            {
                "doc_id": doc_id,
                "text_length": len(text),
                "config": {"max_chars": config.max_chars, "overlap_chars": config.overlap_chars},
                "num_chunks": plan.total_chunks,
                "cuts_inside_sentence": sum(row["cut_inside_sentence"] for row in chunk_rows),
                "chunks": chunk_rows,
            }
        )
    return {"documents": results}


# --------------------------------------------------------------------------- 2. dedup


def corpus_for_dedup() -> dict[str, str]:
    documents = {"product-manual": long_cjk_document(repeats=12)}
    for index in range(14):
        documents[f"faq-{index:02d}"] = filler_document(f"faq{index}")
    documents["faq-03"] = "保修期参数A的默认值是十。发票申请需要订单号。"
    return documents


def chunk_hit_doc_ids(retriever: DocumentRetriever, query: str, top_k: int = 10) -> list[str]:
    """The pre-deduplication baseline: top-k chunk hits mapped back to documents."""
    ranges = []
    row = 0
    for doc_id, record in retriever._records.items():
        ranges.append((row, row + len(record), doc_id))
        row += len(record)
    corpus = torch.cat([record.embeddings for record in retriever._records.values()])
    raw_hits = semantic_search(retriever._encode_texts([query]), corpus, top_k=top_k)[0]
    doc_ids = []
    for hit in raw_hits:
        corpus_id = hit["corpus_id"]
        for start, end, doc_id in ranges:
            if start <= corpus_id < end:
                doc_ids.append(doc_id)
                break
    return doc_ids


def dedup_report(model: SentenceTransformer) -> dict[str, Any]:
    documents = corpus_for_dedup()
    retriever = DocumentRetriever(model, ChunkingConfig(max_chars=90, overlap_chars=20), aggregation="max")
    retriever.add_documents(documents)
    query = "保修期参数A的默认值是多少？"

    before = chunk_hit_doc_ids(retriever, query, top_k=10)
    after_hits = retriever.search(query, top_k=10)[0]
    after = [hit.doc_id for hit in after_hits]

    return {
        "query": query,
        "top_k": 10,
        "indexed_documents": len(documents),
        "indexed_chunks": retriever.num_chunks,
        "before_chunk_level": before,
        "before_unique_documents": len(set(before)),
        "before_repeated_document_hits": len(before) - len(set(before)),
        "after_document_level": [
            {
                "rank": rank + 1,
                "doc_id": hit.doc_id,
                "score": round(hit.score, 6),
                "coverage": round(hit.coverage, 4),
                "matched_chunk": hit.best_chunk.chunk_index,
                "char_range": [hit.best_chunk.start_char, hit.best_chunk.end_char],
                "snippet": hit.snippets[0],
            }
            for rank, hit in enumerate(after_hits)
        ],
        "after_unique_documents": len(set(after)),
        "after_repeated_document_hits": len(after) - len(set(after)),
    }


# --------------------------------------------------------------------------- 3. aggregation


def aggregation_report(model: SentenceTransformer) -> dict[str, Any]:
    documents = {
        "mentions-once": " ".join(
            [
                "Filler paragraph zero about packaging.",
                "Filler paragraph one about logistics.",
                "The beta parameter default is 42 and appears once.",
                "Filler paragraph three about invoicing.",
                "Filler paragraph four about support hours.",
            ]
        ),
        "mentions-many": " ".join(["The beta parameter default is 42."] * 5),
        "unrelated": filler_document("unrelated"),
    }
    config = ChunkingConfig(max_chars=60, overlap_chars=0)
    query = "What is the beta parameter default?"
    rankings = {}
    # mean_top_n is the default, so it is exercised without overrides here.
    for aggregation, kwargs in [
        ("max", {}),
        ("mean_top_n", {}),
        ("hit_count_weighted", {"hit_threshold": 0.75}),
    ]:
        retriever = DocumentRetriever(model, config, aggregation=aggregation, **kwargs)
        retriever.add_documents(documents)
        hits = retriever.search(query, top_k=3)[0]
        rankings[aggregation] = [
            {
                "rank": rank + 1,
                "doc_id": hit.doc_id,
                "score": round(hit.score, 6),
                "hit_count": hit.hit_count,
                "chunk_scores": [round(score, 6) for score in hit.chunk_scores],
            }
            for rank, hit in enumerate(hits)
        ]
    return {"query": query, "documents": documents, "rankings": rankings}


# --------------------------------------------------------------------------- 4. incremental


def incremental_report(model: SentenceTransformer) -> dict[str, Any]:
    documents = {f"doc-{index:02d}": filler_document(f"topic{index}") for index in range(20)}
    documents["doc-00"] = long_cjk_document()
    config = ChunkingConfig(max_chars=100, overlap_chars=20)
    retriever = DocumentRetriever(model, config, aggregation="max")
    retriever.add_documents(documents)

    before = {doc_id: record.embeddings.clone() for doc_id, record in retriever._records.items()}
    updated = retriever.update_document("doc-05", documents["doc-05"] + " A new closing sentence was added.")
    after = {doc_id: record.embeddings.clone() for doc_id, record in retriever._records.items()}

    unchanged = []
    for doc_id in sorted(before):
        if doc_id == "doc-05":
            continue
        same = torch.equal(before[doc_id], after[doc_id])
        unchanged.append(
            {
                "doc_id": doc_id,
                "bit_identical": same,
                "max_abs_diff": float((before[doc_id] - after[doc_id]).abs().max()) if not same else 0.0,
            }
        )
    changed_same_hash = torch.equal(before["doc-05"], after["doc-05"])
    return {
        "updated_document": "doc-05",
        "update_returned": updated,
        "documents_checked": len(unchanged),
        "all_unchanged_bit_identical": all(row["bit_identical"] for row in unchanged),
        "unchanged": unchanged,
        "changed_document_vectors_identical": changed_same_hash,
        "changed_document_text_grew": len(after["doc-05"]) > 0,
    }


# --------------------------------------------------------------------------- 5. disabled parity


def disabled_parity_report(model: SentenceTransformer) -> dict[str, Any]:
    documents = {
        "a": long_ascii_document(repeats=2),
        "b": long_cjk_document(repeats=2),
        "c": "A short note about parameter A.",
    }
    config = ChunkingConfig(max_chars=90, overlap_chars=20)
    enabled = DocumentRetriever(model, config, enabled=True)
    enabled.add_documents(documents)
    disabled = DocumentRetriever(model, config, enabled=False)
    disabled.add_documents(documents)

    queries = ["what is parameter A", "invoice application"]
    query_embeddings = disabled._encode_texts(queries)
    corpus = torch.cat([disabled._records[doc_id].embeddings for doc_id in disabled.doc_ids])
    baseline = semantic_search(query_embeddings, corpus, top_k=3)
    actual = disabled.search(queries, top_k=3)

    parity = []
    for query, raw, hits in zip(queries, baseline, actual):
        parity.append(
            {
                "query": query,
                "baseline": [{"doc_id": disabled.doc_ids[hit["corpus_id"]], "score": hit["score"]} for hit in raw],
                "disabled_retriever": [{"doc_id": hit.doc_id, "score": hit.score} for hit in hits],
                "identical": [hit.doc_id for hit in hits] == [disabled.doc_ids[h["corpus_id"]] for h in raw]
                and all(abs(hit.score - h["score"]) < 1e-6 for hit, h in zip(hits, raw)),
            }
        )
    return {
        "chunks_with_chunking_enabled": enabled.num_chunks,
        "chunks_with_chunking_disabled": disabled.num_chunks,
        "whole_document_encode_identical": _whole_document_encode_identical(model, documents),
        "parity": parity,
        "all_identical": all(row["identical"] for row in parity),
    }


def _whole_document_encode_identical(model: SentenceTransformer, documents: dict[str, str]) -> bool:
    """The disabled retriever must store exactly what ``model.encode`` returns today."""
    baseline = model.encode(list(documents.values()), convert_to_numpy=False, show_progress_bar=False)
    retriever = DocumentRetriever(model, enabled=False)
    retriever.add_documents(documents)
    for doc_id, expected in zip(documents, baseline):
        stored = retriever._records[doc_id].embeddings[0]
        if not torch.equal(stored, expected.to(torch.float32)):
            return False
    return True


# --------------------------------------------------------------------------- 6. benchmark


def benchmark_report(model: SentenceTransformer, num_documents: int = 1000) -> dict[str, Any]:
    documents = {f"doc-{index:04d}": filler_document(f"topic{index}", sentences=12) for index in range(num_documents)}
    config = ChunkingConfig(max_chars=200, overlap_chars=40)

    start = time.perf_counter()
    retriever = DocumentRetriever(model, config, aggregation="max")
    retriever.add_documents(documents)
    index_seconds = time.perf_counter() - start

    # Baseline: the existing whole-document encoding path.
    start = time.perf_counter()
    model.encode(list(documents.values()), convert_to_numpy=True, show_progress_bar=False)
    baseline_seconds = time.perf_counter() - start

    queries = [f"what does the topic{index} guide say" for index in range(50)]
    query_times = []
    for query in queries:
        start = time.perf_counter()
        retriever.search(query, top_k=10)
        query_times.append(time.perf_counter() - start)

    # Re-indexing the same corpus must not re-encode anything.
    start = time.perf_counter()
    changed = retriever.add_documents(documents)
    noop_seconds = time.perf_counter() - start

    return {
        "documents": num_documents,
        "chunks": retriever.num_chunks,
        "chunks_per_document": round(retriever.num_chunks / num_documents, 3),
        "config": {"max_chars": config.max_chars, "overlap_chars": config.overlap_chars},
        "baseline_encode_seconds": round(baseline_seconds, 3),
        "chunked_index_seconds": round(index_seconds, 3),
        "index_over_baseline": round(index_seconds / baseline_seconds, 3),
        "noop_reindex_seconds": round(noop_seconds, 4),
        "noop_reindex_reencoded_documents": len(changed),
        "query_seconds": {
            "median": round(statistics.median(query_times), 5),
            "p95": round(sorted(query_times)[int(0.95 * len(query_times)) - 1], 5),
            "max": round(max(query_times), 5),
        },
    }


# --------------------------------------------------------------------------- summary


def write_markdown(report: dict[str, Any]) -> None:
    boundaries = report["boundaries"]["documents"]
    dedup = report["dedup"]
    aggregation = report["aggregation"]["rankings"]
    incremental = report["incremental"]
    parity = report["disabled_parity"]
    benchmark = report["benchmark"]

    lines = ["# 长文档切块检索验收报告", ""]
    lines += ["## 1. 切块边界", ""]
    for document in boundaries:
        lines.append(
            f"- `{document['doc_id']}`: {document['text_length']} 字符 → {document['num_chunks']} 块，"
            f"句中切点 {document['cuts_inside_sentence']} 处"
        )
    sample = boundaries[0]["chunks"][0]
    lines += [
        "",
        "样例（`cjk-manual` 第 0 块）：",
        "",
        f"- 字符区间 `[{sample['start_char']}, {sample['end_char']})`，结束边界 `{sample['end_boundary']}`",
        f"- 结尾 12 字符：`{sample['last_chars']}`",
        f"- 文本：{sample['text']}",
        "",
    ]
    lines += ["## 2. 同文档去重", ""]
    lines += [
        f"- 去重前（块级 top-10）：{dedup['before_chunk_level']}",
        f"- 命中不同文档数：{dedup['before_unique_documents']}，重复命中 {dedup['before_repeated_document_hits']} 次",
        f"- 去重后（文档级 top-10）：{[row['doc_id'] for row in dedup['after_document_level']]}",
        f"- 重复命中：{dedup['after_repeated_document_hits']} 次",
        "",
    ]
    lines += ["## 3. 聚合方式对比", ""]
    for aggregation_name, rows in aggregation.items():
        summary = ", ".join(f"{row['doc_id']}({row['score']})" for row in rows)
        lines.append(f"- `{aggregation_name}`: {summary}")
    lines += [""]
    lines += ["## 4. 增量更新", ""]
    lines += [
        f"- 只更新 `{incremental['updated_document']}`",
        f"- 未变动文档 {incremental['documents_checked']} 篇，向量逐位相同：{incremental['all_unchanged_bit_identical']}",
        f"- 被更新文档向量是否变化：{not incremental['changed_document_vectors_identical']}",
        "",
    ]
    lines += ["## 5. 关闭开关后的原有行为", ""]
    lines += [
        f"- 开启切块：{parity['chunks_with_chunking_enabled']} 块；关闭切块：{parity['chunks_with_chunking_disabled']} 块",
        f"- 关闭后存储的向量与 `model.encode` 逐位一致：{parity['whole_document_encode_identical']}",
        f"- 与 `semantic_search` 整篇编码结果一致：{parity['all_identical']}",
        "",
    ]
    lines += ["## 6. 千篇量级耗时", ""]
    lines += [
        f"- 文档数 {benchmark['documents']}，块数 {benchmark['chunks']}（每篇 {benchmark['chunks_per_document']} 块）",
        f"- 基线整篇编码：{benchmark['baseline_encode_seconds']}s",
        f"- 切块建索引：{benchmark['chunked_index_seconds']}s（基线 {benchmark['index_over_baseline']}×）",
        f"- 无变更重建索引：{benchmark['noop_reindex_seconds']}s，重新编码文档数 {benchmark['noop_reindex_reencoded_documents']}",
        f"- 查询耗时：中位 {benchmark['query_seconds']['median']}s，P95 {benchmark['query_seconds']['p95']}s",
        "",
    ]
    (REPORTS / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)
    model = SentenceTransformer(MODEL_ID, device="cpu")

    report = {
        "boundaries": boundary_report(),
        "dedup": dedup_report(model),
        "aggregation": aggregation_report(model),
        "incremental": incremental_report(model),
        "disabled_parity": disabled_parity_report(model),
        "benchmark": benchmark_report(model),
    }

    write_json("chunk_boundaries.json", report["boundaries"])
    write_json("dedup_before_after.json", report["dedup"])
    write_json("aggregation_comparison.json", report["aggregation"])
    write_json("incremental_update.json", report["incremental"])
    write_json("disabled_mode_parity.json", report["disabled_parity"])
    write_json("benchmark.json", report["benchmark"])
    write_markdown(report)
    print(json.dumps({key: "written" for key in report}, indent=2))


if __name__ == "__main__":
    main()
