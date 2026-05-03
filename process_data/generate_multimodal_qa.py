import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


FAULT_SYSTEM_PROMPT = (
    "You are a seismic interpretation assistant. Answer from the provided seismic "
    "image, fault metadata, mask metadata, and report context only. Use [interp] "
    "for report interpretation, [fault] for fault interpretation, and [seg] when "
    "the answer requires the binary fault segmentation target."
)

TASK_TOKENS = {
    "fault_interpretation": "[fault]",
    "fault_segmentation": "[seg]",
    "seismic_interpretation": "[interp]",
}

FAULT_CONTEXT_TERMS = (
    "fault",
    "faults",
    "fault block",
    "fault reactivation",
    "caprock",
    "seal",
    "gn1101",
    "seismic",
    "storage complex",
    "alpha",
    "beta",
    "smeaheia",
    "troll",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def extract_json_array(text: str) -> list[dict[str, str]]:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)

    parsed = None
    parse_error: Exception | None = None
    for candidate in candidates:
        candidate = candidate.strip()
        spans = [candidate]
        array_match = re.search(r"\[[\s\S]*\]", candidate)
        if array_match:
            spans.insert(0, array_match.group(0))
        object_match = re.search(r"\{[\s\S]*\}", candidate)
        if object_match:
            spans.append(object_match.group(0))

        for span in spans:
            try:
                parsed = json.loads(span)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc
        if parsed is not None:
            break

    if parsed is None:
        preview = text.replace("\n", " ")[:240]
        raise ValueError(f"LLM response did not contain parseable JSON. Preview: {preview!r}") from parse_error
    if isinstance(parsed, dict):
        if isinstance(parsed.get("items"), list):
            parsed = parsed["items"]
        elif isinstance(parsed.get("qa_pairs"), list):
            parsed = parsed["qa_pairs"]
        elif "question" in parsed and "answer" in parsed:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("LLM response JSON was not a list of question/answer objects")
    records = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            records.append({"question": question, "answer": answer})
    if not records:
        raise ValueError("LLM response contained no valid question/answer objects")
    return records


def json_only_prompt(prompt: str) -> str:
    return (
        "You are generating training data. Return valid JSON only.\n"
        "Do not include markdown fences, prose, explanations, comments, or headings.\n"
        'The entire response must be a JSON array like: [{"question":"...","answer":"..."}].\n'
        "Every answer must be grounded only in the supplied metadata/context.\n\n"
        f"{prompt}\n\n"
        "Return only the JSON array now."
    )


def media_context(record: dict[str, Any], max_chars: int) -> str:
    before = [item.get("text", "") for item in record.get("nearby_text_before", [])]
    after = [item.get("text", "") for item in record.get("nearby_text_after", [])]
    return compact_text(
        [
            f"Caption: {record.get('caption', '')}",
            "Section: " + " > ".join(record.get("section_path", [])),
            "Before:\n" + "\n".join(before),
            "After:\n" + "\n".join(after),
            "Same page:\n" + record.get("same_page_text", ""),
        ],
        max_chars=max_chars,
    )


def collect_media_records(interpretation_root: Path) -> list[dict[str, Any]]:
    records = []
    for manifest_path in sorted(interpretation_root.glob("*/manifest.jsonl")):
        records.extend(read_jsonl(manifest_path))
    return records


def collect_report_chunks(interpretation_root: Path, max_chars: int) -> str:
    selected: list[str] = []
    for chunks_path in sorted(interpretation_root.glob("*/chunks.jsonl")):
        for chunk in read_jsonl(chunks_path):
            text = chunk.get("text", "")
            lowered = text.lower()
            if any(term in lowered for term in FAULT_CONTEXT_TERMS):
                section = " > ".join(chunk.get("section_path", []))
                selected.append(f"{chunk.get('doc_id', chunks_path.parent.name)} | {section}\n{text}")
    return compact_text(selected, max_chars=max_chars)


class SyntheticQAGenerator:
    def __init__(self, provider: str, model: str | None, temperature: float):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.llm = None

    def setup(self, report_context: str) -> None:
        try:
            from llama_index.core import Settings
        except ImportError as exc:
            raise ImportError(
                "Synthetic QA requires LlamaIndex. Install it with: uv sync --extra synthetic"
            ) from exc

        if self.provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is required when --llm-provider anthropic is used.")
            from llama_index.llms.anthropic import Anthropic

            self.llm = Anthropic(
                model=self.model or "claude-sonnet-4-0",
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unsupported --llm-provider: {self.provider}")

        Settings.llm = self.llm

    def generate(self, prompt: str) -> list[dict[str, str]]:
        if self.llm is None:
            raise RuntimeError("SyntheticQAGenerator.setup() must be called before generate().")
        response = self.llm.complete(json_only_prompt(prompt))
        return extract_json_array(str(response))


def llama_fault_prompt(record: dict[str, Any], report_context: str, count: int) -> str:
    payload = {
        "survey": record.get("survey"),
        "slice_type": record.get("slice_type"),
        "slice_index": record.get("slice_index"),
        "image_shape": record.get("image_shape"),
        "mask_shape": record.get("mask_shape"),
        "has_fault": record.get("has_fault"),
        "fault_names": record.get("fault_names", []),
        "fault_point_count": record.get("fault_point_count", 0),
        "fault_source": record.get("fault_source"),
    }
    return (
        "Create supervised synthetic QA for a seismic VLM fault-interpretation dataset.\n"
        f"Return exactly {count} JSON objects as a JSON array. Each object must have only "
        '"question" and "answer" keys.\n'
        "The task is fault interpretation, not a plain classification label. The question should ask "
        "the model to explain fault evidence, geological meaning, or confidence from the seismic inline. "
        "The answer must stay grounded in the metadata and report context. Do not invent coordinates, "
        "formations, or fault names that are not supplied.\n\n"
        f"Fault metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Relevant extracted report context:\n{report_context}"
    )


def llama_media_prompt(record: dict[str, Any], context: str, count: int) -> str:
    payload = {
        "doc_id": record.get("doc_id"),
        "element_id": record.get("element_id"),
        "media_type": record.get("type"),
        "page": record.get("page"),
        "section_path": record.get("section_path", []),
        "caption": record.get("caption", ""),
    }
    return (
        "Create supervised synthetic QA for a seismic report interpretation dataset.\n"
        f"Return exactly {count} JSON objects as a JSON array. Each object must have only "
        '"question" and "answer" keys.\n'
        "Questions should ask for interpretation of the extracted figure/table. Answers must be grounded "
        "in the caption and nearby report text, and should explain why the visual matters for seismic or "
        "subsurface interpretation.\n\n"
        f"Media metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Extracted visual context:\n{context}"
    )


def qa_record(
    *,
    source: dict[str, Any],
    task: str,
    dataset: str,
    question: str,
    answer: str,
    split: str,
    image_path: str | None,
    mask_path: str | None,
    metadata: dict[str, Any],
    variant: int,
    synthetic: bool,
) -> dict[str, Any]:
    token = TASK_TOKENS[task]
    record_id = stable_id(dataset, task, source.get("slice_index"), source.get("doc_id"), source.get("element_id"), variant)
    return {
        "id": record_id,
        "split": split,
        "dataset": dataset,
        "task": task,
        "task_token": token,
        "modality": "seismic_image_mask" if mask_path else "report_media_text",
        "image_path": image_path,
        "mask_path": mask_path,
        "question": question,
        "answer": answer,
        "messages": messages(question, image_path, token) + [{"role": "assistant", "content": answer}],
        "metadata": {
            **metadata,
            "synthetic": synthetic,
        },
    }


def build_fault_records(
    info_path: Path,
    val_fraction: float,
    report_context: str,
    synthetic_generator: SyntheticQAGenerator,
    synthetic_count: int,
    max_records: int | None,
) -> list[dict[str, Any]]:
    records = []
    fault_records = read_jsonl(info_path)
    if max_records is not None:
        fault_records = fault_records[:max_records]

    for record in tqdm(fault_records, desc="Fault interpretation QA"):
        image_path = record["image_path"]
        mask_path = record["mask_path"]
        slice_id = record["slice_index"]
        split = split_name(f"fault:{slice_id}", val_fraction)
        metadata = {
            "survey": record.get("survey"),
            "slice_type": record.get("slice_type"),
            "slice_index": slice_id,
            "has_fault": bool(record.get("has_fault")),
            "fault_names": record.get("fault_names", []),
            "fault_point_count": record.get("fault_point_count", 0),
            "image_shape": record.get("image_shape"),
            "mask_shape": record.get("mask_shape"),
        }

        pairs = synthetic_generator.generate(llama_fault_prompt(record, report_context, synthetic_count))

        for variant, pair in enumerate(pairs, start=1):
            records.append(qa_record(
                source=record,
                task="fault_interpretation",
                dataset="fault_interpretation",
                question=pair["question"],
                answer=pair["answer"],
                split=split,
                image_path=image_path,
                mask_path=mask_path,
                metadata=metadata,
                variant=variant,
                synthetic=True,
            ))

        records.append(qa_record(
            source=record,
            task="fault_segmentation",
            dataset="fault_interpretation",
            question="Return the binary fault segmentation target for this seismic section.",
            answer=(
                f"[seg] Use the binary fault mask at {mask_path} as the segmentation target."
                if record.get("has_fault")
                else "[seg] The segmentation target is an empty binary mask because no mapped fault pixels are present."
            ),
            split=split,
            image_path=image_path,
            mask_path=mask_path,
            metadata=metadata,
            variant=999,
            synthetic=False,
        ))

    return records


def build_interpretation_records(
    interpretation_root: Path,
    val_fraction: float,
    max_context_chars: int,
    synthetic_generator: SyntheticQAGenerator,
    synthetic_count: int,
    max_records: int | None,
) -> list[dict[str, Any]]:
    records = []
    media_records = collect_media_records(interpretation_root)
    if max_records is not None:
        media_records = media_records[:max_records]

    for record in tqdm(media_records, desc="Seismic interpretation QA"):
        image_path = record.get("image_path")
        context = media_context(record, max_context_chars)
        key = f"interpretation:{record.get('doc_id')}:{record.get('element_id')}"
        split = split_name(key, val_fraction)
        metadata = {
            "doc_id": record.get("doc_id"),
            "element_id": record.get("element_id"),
            "media_type": record.get("type"),
            "page": record.get("page"),
            "section_path": record.get("section_path", []),
            "caption": record.get("caption"),
        }

        pairs = synthetic_generator.generate(llama_media_prompt(record, context, synthetic_count))

        for variant, pair in enumerate(pairs, start=1):
            records.append(qa_record(
                source=record,
                task="seismic_interpretation",
                dataset="seismic_interpretation",
                question=pair["question"],
                answer=pair["answer"],
                split=split,
                image_path=image_path,
                mask_path=None,
                metadata=metadata,
                variant=variant,
                synthetic=True,
            ))

    return records


def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    write_jsonl(output_dir / "all.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", [record for record in records if record["split"] == "train"])
    write_jsonl(output_dir / "val.jsonl", [record for record in records if record["split"] == "val"])

    summary: dict[str, Any] = {
        "total": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "val": sum(record["split"] == "val" for record in records),
        "synthetic": sum(bool(record.get("metadata", {}).get("synthetic")) for record in records),
        "by_dataset": {},
        "by_task": {},
    }
    for record in records:
        summary["by_dataset"][record["dataset"]] = summary["by_dataset"].get(record["dataset"], 0) + 1
        summary["by_task"][record["task"]] = summary["by_task"].get(record["task"], 0) + 1

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multimodal QA from fault masks and extracted seismic interpretation report data."
    )
    parser.add_argument("--fault-info", type=Path, default=Path("process_data/fault_detection/data/info.jsonl"))
    parser.add_argument("--interpretation-root", type=Path, default=Path("process_data/interpretation"))
    parser.add_argument("--output-dir", type=Path, default=Path("process_data/multimodal_qa"))
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-report-context-chars", type=int, default=12000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--fault-only", action="store_true")
    parser.add_argument("--interpretation-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use Anthropic through LlamaIndex to synthesize QA pairs.")
    parser.add_argument("--llm-provider", choices=["anthropic"], default="anthropic")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--synthetic-per-record", type=int, default=2)
    parser.add_argument("--max-fault-records", type=int, default=0)
    parser.add_argument("--max-interpretation-records", type=int, default=0)
    args = parser.parse_args()

    if not args.synthetic:
        raise ValueError("Synthetic QA generation requires the Anthropic API. Pass --synthetic.")

    report_context = collect_report_chunks(args.interpretation_root, args.max_report_context_chars)
    synthetic_generator = SyntheticQAGenerator(
        provider=args.llm_provider,
        model=args.llm_model,
        temperature=args.llm_temperature,
    )
    synthetic_generator.setup(report_context)

    records: list[dict[str, Any]] = []
    if not args.interpretation_only:
        records.extend(build_fault_records(
            info_path=args.fault_info,
            val_fraction=args.val_fraction,
            report_context=report_context,
            synthetic_generator=synthetic_generator,
            synthetic_count=args.synthetic_per_record,
            max_records=args.max_fault_records or None,
        ))

    if not args.fault_only:
        records.extend(build_interpretation_records(
            interpretation_root=args.interpretation_root,
            val_fraction=args.val_fraction,
            max_context_chars=args.max_context_chars,
            synthetic_generator=synthetic_generator,
            synthetic_count=args.synthetic_per_record,
            max_records=args.max_interpretation_records or None,
        ))

    save_outputs(records, args.output_dir)
    print(f"wrote {len(records)} QA records to {args.output_dir}")


if __name__ == "__main__":
    main()
