import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


FAULT_SYSTEM_PROMPT = (
    "You are a seismic interpretation assistant. Answer from the provided seismic "
    "image, fault metadata, and report context only. Use [interp] for report "
    "interpretation, [fault] for fault QA, and [seg] when the answer requires "
    "the fault segmentation target."
)

TASK_TOKENS = {
    "fault_presence": "[fault]",
    "fault_names": "[fault]",
    "fault_segmentation": "[seg]",
    "interpret_caption": "[interp]",
    "interpret_context": "[interp]",
    "llamaindex_interpretation": "[interp]",
}

INTERPRETATION_QUESTION_PROMPT = """
Generate geoscience training questions for the provided Smeaheia report figure/table.
The questions must be answerable from the caption and nearby report context.
Prefer questions about seismic interpretation, faults, structure, storage context,
well evidence, or map/section meaning. Avoid generic document-summary questions.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def split_name(key: str, val_fraction: float) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_fraction else "train"


def compact_text(parts: list[str], max_chars: int) -> str:
    text = "\n".join(part.strip() for part in parts if part and part.strip())
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text[:max_chars].rstrip()


def task_prompt(task_token: str, question: str) -> str:
    return f"{task_token} {question}" if task_token else question


def messages(question: str, image_path: str | None, task_token: str = "") -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "text", "text": task_prompt(task_token, question)}]
    if image_path:
        content.append({"type": "image_path", "image_path": image_path})
    return [
        {"role": "system", "content": FAULT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_fault_qa(info_path: Path, val_fraction: float) -> list[dict[str, Any]]:
    records = []
    for record in tqdm(read_jsonl(info_path), desc="Fault QA"):
        image_path = record["image_path"]
        mask_path = record["mask_path"]
        slice_id = record["slice_index"]
        has_fault = bool(record["has_fault"])
        fault_names = record.get("fault_names", [])
        split = split_name(f"fault:{slice_id}", val_fraction)

        base_metadata = {
            "survey": record.get("survey"),
            "slice_type": record.get("slice_type"),
            "slice_index": slice_id,
            "has_fault": has_fault,
            "fault_names": fault_names,
            "fault_point_count": record.get("fault_point_count", 0),
        }

        qa_specs = [
            (
                "fault_presence",
                "Does this inline seismic section contain mapped fault evidence? Answer yes or no and briefly justify from the provided metadata.",
                (
                    f"Yes. This inline has mapped fault evidence with {record.get('fault_point_count', 0)} fault-stick points."
                    if has_fault
                    else "No. This inline has no mapped fault mask pixels in the generated label."
                ),
            ),
            (
                "fault_segmentation",
                "Return the fault segmentation target for this seismic section. Use the [seg] token.",
                (
                    f"[seg] Use the binary fault mask at {mask_path} as the segmentation target."
                    if has_fault
                    else "[seg] The segmentation target is an empty binary mask because no mapped fault pixels are present."
                ),
            ),
        ]

        if fault_names:
            qa_specs.append((
                "fault_names",
                "Which interpreted fault names are associated with this inline section?",
                "The associated interpreted fault names are: " + ", ".join(fault_names) + ".",
            ))

        for task, question, answer in qa_specs:
            token = TASK_TOKENS[task]
            records.append({
                "id": stable_id("fault", task, slice_id),
                "split": split,
                "dataset": "fault_detection",
                "task": task,
                "task_token": token,
                "modality": "seismic_image_mask",
                "image_path": image_path,
                "mask_path": mask_path,
                "question": question,
                "answer": answer,
                "messages": messages(question, image_path, token) + [{"role": "assistant", "content": answer}],
                "metadata": base_metadata,
            })

    return records


def media_context(record: dict[str, Any], max_chars: int) -> str:
    before = [item.get("text", "") for item in record.get("nearby_text_before", [])]
    after = [item.get("text", "") for item in record.get("nearby_text_after", [])]
    return compact_text(
        [
            f"Caption: {record.get('caption', '')}",
            "Section: " + " > ".join(record.get("section_path", [])),
            "Before:\n" + "\n".join(before),
            "After:\n" + "\n".join(after),
        ],
        max_chars=max_chars,
    )


def collect_media_records(interpretation_root: Path) -> list[dict[str, Any]]:
    records = []
    for manifest_path in sorted(interpretation_root.glob("*/manifest.jsonl")):
        records.extend(read_jsonl(manifest_path))
    return records


def build_template_interpretation_qa(
    interpretation_root: Path,
    val_fraction: float,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    qa_records = []
    for record in tqdm(collect_media_records(interpretation_root), desc="Template interpretation QA"):
        image_path = record.get("image_path")
        caption = record.get("caption", "")
        context = media_context(record, max_context_chars)
        key = f"interpretation:{record.get('doc_id')}:{record.get('element_id')}"
        split = split_name(key, val_fraction)
        metadata = {
            "doc_id": record.get("doc_id"),
            "element_id": record.get("element_id"),
            "media_type": record.get("type"),
            "page": record.get("page"),
            "section_path": record.get("section_path", []),
            "caption": caption,
        }

        qa_specs = [
            (
                "interpret_caption",
                "What does this report figure or table show, and why is it useful for seismic interpretation?",
                f"{caption}\n\nRelevant report context:\n{context}",
            ),
            (
                "interpret_context",
                "Explain the geological or seismic interpretation context for this visual using the caption and nearby report text.",
                context,
            ),
        ]

        for task, question, answer in qa_specs:
            token = TASK_TOKENS[task]
            qa_records.append({
                "id": stable_id("interpretation", task, record.get("doc_id"), record.get("element_id")),
                "split": split,
                "dataset": "seismic_interpretation",
                "task": task,
                "task_token": token,
                "modality": "report_media_text",
                "image_path": image_path,
                "mask_path": None,
                "question": question,
                "answer": answer,
                "messages": messages(question, image_path, token) + [{"role": "assistant", "content": answer}],
                "metadata": metadata,
            })

    return qa_records


def make_llamaindex_llm(model: str, temperature: float):
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex OpenAI integration is not installed. Install project dependencies "
            "and set OPENAI_API_KEY, or run with --generator template."
        ) from exc

    return OpenAI(model=model, temperature=temperature)


def llamaindex_pairs_for_record(
    record: dict[str, Any],
    llm: Any,
    questions_per_media: int,
    max_context_chars: int,
) -> list[tuple[str, str]]:
    try:
        from llama_index.core import Document
        from llama_index.core.llama_dataset.generator import RagDatasetGenerator
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex is not installed. Install project dependencies or run with --generator template."
        ) from exc

    document = Document(
        text=media_context(record, max_context_chars),
        metadata={
            "doc_id": record.get("doc_id"),
            "element_id": record.get("element_id"),
            "image_path": record.get("image_path"),
            "caption": record.get("caption", ""),
        },
    )
    generator = RagDatasetGenerator.from_documents(
        [document],
        llm=llm,
        num_questions_per_chunk=questions_per_media,
        question_gen_query=INTERPRETATION_QUESTION_PROMPT,
        show_progress=False,
    )
    dataset = generator.generate_dataset_from_nodes(num=questions_per_media)
    data = dataset.model_dump() if hasattr(dataset, "model_dump") else dataset.dict()

    if "examples" in data:
        pairs = []
        for example in data["examples"]:
            question = example.get("query") or example.get("question")
            answer = example.get("reference_answer") or example.get("answer") or ""
            if question and answer:
                pairs.append((question, answer))
        return pairs

    queries = data.get("queries", {})
    responses = data.get("responses", {})
    return [(query, responses.get(query_id, "")) for query_id, query in queries.items() if responses.get(query_id)]


def build_llamaindex_interpretation_qa(
    interpretation_root: Path,
    val_fraction: float,
    model: str,
    temperature: float,
    questions_per_media: int,
    max_context_chars: int,
) -> list[dict[str, Any]]:
    llm = make_llamaindex_llm(model=model, temperature=temperature)
    qa_records = []

    for record in tqdm(collect_media_records(interpretation_root), desc="LlamaIndex interpretation QA"):
        key = f"interpretation:{record.get('doc_id')}:{record.get('element_id')}"
        split = split_name(key, val_fraction)
        metadata = {
            "doc_id": record.get("doc_id"),
            "element_id": record.get("element_id"),
            "media_type": record.get("type"),
            "page": record.get("page"),
            "section_path": record.get("section_path", []),
            "caption": record.get("caption", ""),
            "generator": "llama-index",
            "llm_model": model,
        }

        for idx, (question, answer) in enumerate(
            llamaindex_pairs_for_record(record, llm, questions_per_media, max_context_chars),
            start=1,
        ):
            token = TASK_TOKENS["llamaindex_interpretation"]
            qa_records.append({
                "id": stable_id("interpretation", "llama-index", record.get("doc_id"), record.get("element_id"), idx),
                "split": split,
                "dataset": "seismic_interpretation",
                "task": "llamaindex_interpretation",
                "task_token": token,
                "modality": "report_media_text",
                "image_path": record.get("image_path"),
                "mask_path": None,
                "question": question,
                "answer": answer,
                "messages": messages(question, record.get("image_path"), token) + [{"role": "assistant", "content": answer}],
                "metadata": metadata,
            })

    return qa_records


def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    write_jsonl(output_dir / "all.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", [record for record in records if record["split"] == "train"])
    write_jsonl(output_dir / "val.jsonl", [record for record in records if record["split"] == "val"])

    summary = {
        "total": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "val": sum(record["split"] == "val" for record in records),
        "by_dataset": {},
        "by_task": {},
    }
    for record in records:
        summary["by_dataset"][record["dataset"]] = summary["by_dataset"].get(record["dataset"], 0) + 1
        summary["by_task"][record["task"]] = summary["by_task"].get(record["task"], 0) + 1

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multimodal QA pairs from fault info.jsonl and interpretation manifest.jsonl files."
    )
    parser.add_argument("--fault-info", type=Path, default=Path("process_data/fault_detection/data/info.jsonl"))
    parser.add_argument("--interpretation-root", type=Path, default=Path("process_data/interpretation"))
    parser.add_argument("--output-dir", type=Path, default=Path("process_data/multimodal_qa"))
    parser.add_argument("--generator", choices=["llama-index", "template"], default="llama-index")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--questions-per-media", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--fault-only", action="store_true")
    parser.add_argument("--interpretation-only", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if not args.interpretation_only:
        records.extend(build_fault_qa(args.fault_info, args.val_fraction))

    if not args.fault_only:
        if args.generator == "llama-index":
            records.extend(
                build_llamaindex_interpretation_qa(
                    interpretation_root=args.interpretation_root,
                    val_fraction=args.val_fraction,
                    model=args.llm_model,
                    temperature=args.temperature,
                    questions_per_media=args.questions_per_media,
                    max_context_chars=args.max_context_chars,
                )
            )
        else:
            records.extend(
                build_template_interpretation_qa(
                    interpretation_root=args.interpretation_root,
                    val_fraction=args.val_fraction,
                    max_context_chars=args.max_context_chars,
                )
            )

    save_outputs(records, args.output_dir)
    print(f"wrote {len(records)} QA records to {args.output_dir}")


if __name__ == "__main__":
    main()
