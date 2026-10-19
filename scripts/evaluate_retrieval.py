"""Evaluate the real saved corpus without modifying it or calling an LLM."""

import argparse
import json
import os
import pickle
import sys
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from pydantic import TypeAdapter  # noqa: E402
from rag_evaluation import (  # noqa: E402
    RetrievalCase,
    choose_dev_configuration,
    evaluate_snapshot,
    validate_dataset,
)
from retrieval_utils import LexicalIndex, chunk_text, context_excerpt  # noqa: E402


def main():
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "reconfigure", None):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evidence")
    parser.add_argument("--backend", choices=["lexical", "semantic"], default="lexical")
    parser.add_argument("--allow-model-load", action="store_true")
    args = parser.parse_args()
    if args.backend == "semantic" and not args.allow_model_load:
        parser.error("语义模式可能加载或下载 BGE，需要 --allow-model-load")
    try:
        dataset = ROOT / "evaluation/retrieval_cases.json"
        cases = TypeAdapter(list[RetrievalCase]).validate_json(
            dataset.read_text(encoding="utf-8")
        )
        catalog_path = ROOT / "embedding/file_info.pkl"
        # Only the repository's existing trusted pickle is loaded, not user uploads.
        with catalog_path.open("rb") as stream:
            mapping = pickle.load(stream)
        documents = [(text, source) for text, source in mapping.items()]
        validate_dataset(cases, documents)
        dev = [case for case in cases if case.split == "dev"]
        test = [case for case in cases if case.split == "test"]
        retrievers = {}
        build_times = {}
        started = perf_counter()
        retrievers["saved_chunks"] = LexicalIndex(documents)
        build_times["saved_chunks"] = (perf_counter() - started) * 1000
        rechunked = [
            (part, f"{source}#subchunk{position}")
            for text, source in documents
            for position, part in enumerate(
                chunk_text(text, max_characters=600, overlap_characters=80), 1
            )
        ]
        started = perf_counter()
        retrievers["rechunked_600_80"] = LexicalIndex(rechunked)
        build_times["rechunked_600_80"] = (perf_counter() - started) * 1000
        if args.backend == "semantic":
            os.environ["RAG_BACKEND"] = "semantic"
            from embed import TextEmbedding

            semantic = TextEmbedding.load_with_file_info(str(ROOT / "embedding"))
            if semantic.get_status()["backend"] != "semantic":
                raise ValueError("没有有效语义后端，不能把关键词结果当语义结果")
            retrievers = {"bge_saved_chunks": semantic}
            build_times = {}
        selected, all_experiments = {}, {}
        for name, index in retrievers.items():
            snapshots, durations = {}, {}

            def search(case):
                start = perf_counter()
                result = (
                    index.search_similar_texts(
                        case.question, top_n=5, min_similarity=0.0
                    )
                    if name == "bge_saved_chunks"
                    else index.search(case.question, top_n=5, min_score=0.0)
                )
                durations[case.id] = (perf_counter() - start) * 1000
                return result

            for case in dev:
                snapshots[case.id] = search(case)
            experiments = []
            for k, threshold in product((1, 3, 5), (0.05, 0.15, 0.30, 0.50)):
                metrics = evaluate_snapshot(
                    dev, snapshots, top_k=k, threshold=threshold
                )
                experiments.append(
                    {
                        "split": "dev",
                        "top_k": k,
                        "threshold": threshold,
                        "metrics": {
                            key: value
                            for key, value in metrics.items()
                            if key != "results"
                        },
                    }
                )
            best = choose_dev_configuration(experiments)
            for case in test:
                snapshots[case.id] = search(case)
            delivered = {
                case.id: [
                    (context_excerpt(text, case.question), score, source)
                    for text, score, source in snapshots[case.id]
                ]
                for case in test
            }
            selected[name] = {
                "top_k": best["top_k"],
                "threshold": best["threshold"],
                "dev_metrics": evaluate_snapshot(
                    dev, snapshots, top_k=best["top_k"], threshold=best["threshold"]
                ),
                "test_metrics": evaluate_snapshot(
                    test, snapshots, top_k=best["top_k"], threshold=best["threshold"]
                ),
                "test_tool_excerpt_metrics": evaluate_snapshot(
                    test, delivered, top_k=best["top_k"], threshold=best["threshold"]
                ),
                "query_elapsed_ms": durations,
            }
            all_experiments[name] = experiments
        payload = {
            "scope": "retrieval_evidence_only",
            "backend": args.backend,
            "model_load_allowed": args.allow_model_load,
            "generation_calls": False,
            "source_count": len(
                {source.split("#chunk", 1)[0] for _, source in documents}
            ),
            "saved_chunk_count": len(documents),
            "rechunked_count": len(rechunked),
            "case_count": len(cases),
            "dev_count": len(dev),
            "test_count": len(test),
            "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
            "catalog_sha256": sha256(catalog_path.read_bytes()).hexdigest(),
            "lexical_index_build_ms": build_times,
            "selected_by_mode": selected,
            "dev_experiments": all_experiments,
            "selection_rule": "dev evidence hit and negative empty rate equally weighted; ties prefer evidence hit, empty rate, smaller k and threshold",
        }
        directory = args.output_dir.resolve() / f"retrieval-{uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            "# SolarAI-X 检索评测",
            "",
            f"{payload['source_count']} 份现有来源，{len(documents)} 个原片段；{len(cases)} 道作者编写问题，开发/测试各 {len(dev)} 道。",
            "",
            "| 模式 | k | 阈值 | 测试来源命中 | 测试证据命中 | 证据MRR | 无答案空检索 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, result in selected.items():
            metrics = result["test_metrics"]
            lines.append(
                f"| {name} | {result['top_k']} | {result['threshold']:.2f} | {metrics['source_recall_at_k']:.1%} | {metrics['evidence_hit_at_k']:.1%} | {metrics['evidence_mrr_at_k']:.3f} | {metrics['unanswerable_empty_rate']:.1%} |"
            )
        lines.extend(
            [
                "",
                "工具输出每片段最多保留 4000 字符；长片段选取查询词覆盖最多的连续原文窗口。",
            ]
        )
        for name, result in selected.items():
            excerpt_metrics = result["test_tool_excerpt_metrics"]
            lines.append(
                f"- {name}：实际输出摘要窗口的证据命中为 {excerpt_metrics['evidence_hit_at_k']:.1%}，与完整片段指标分别记录，不评价模型回答。"
            )
        for name, result in selected.items():
            lines.extend(["", f"## {name} 需检查题目", ""])
            for row in result["test_metrics"]["results"]:
                if row["evidence_hit"] is False or row["unanswerable_empty"] is False:
                    lines.append(f"- {row['id']}：{row['question']}")
        lines.extend(
            [
                "",
                "## 范围",
                "",
                "按开发组选择参数；测试组不参与选择。这是作者根据现有手册设计的小题集，不是外部业务测试集。",
                "原文锚点检查只评估检索证据，不等于生成回答准确率、引用正确率或模型拒答率。",
                "重切片仅在评测内存中进行，不覆盖原 PKL、不重新训练模型、不改变线上阈值。",
                "英文手册包含提取噪声；中文问题属于跨语言挑战，关键词降级模式不等于跨语言语义模型。",
                "现有文本去重可能丢弃相同内容的其他来源，本次按当前保存的来源映射评估。",
                "检索计时不包含索引构建、模型加载、回答生成或网络传输，不用于宣称业务响应延迟。",
                "默认完全离线；可选语义模式须显式允许加载模型。本报告不评价光伏预测模型科学精度。",
                "",
            ]
        )
        (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"评测完成：{len(cases)} 道问题，报告：{directory}")
    for name, result in selected.items():
        print(name, "证据命中：", result["test_metrics"]["evidence_hit_at_k"])


if __name__ == "__main__":
    main()
