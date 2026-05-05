import argparse
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

from pydantic.warnings import UnsupportedFieldAttributeWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience only
    load_dotenv = None

from distilabel.models.llms import AnthropicLLM


FAULT_SYSTEM_PROMPT = (
    "You are a seismic interpretation assistant. Answer from the provided seismic "
    "image metadata, fault metadata, mask metadata, and report context only. Use "
    "[interp] for report interpretation, [fault] for fault interpretation, and "
    "[seg] when the answer requires the binary fault segmentation target."
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

REQUIRED_FIELDS = {
    "id",
    "split",
    "dataset",
    "task",
    "task_token",
    "modality",
    "image_path",
    "mask_path",
    "question",
    "answer",
    "messages",
    "metadata",
}


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


def fault_prompt(record: dict[str, Any], report_context: str, count: int) -> str:
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
        "The task is fault interpretation, not plain classification. Questions should ask "
        "the model to explain fault evidence, geological meaning, uncertainty, or confidence "
        "from the seismic inline. Answers must stay grounded in metadata/context. Do not "
        "invent coordinates, formations, or fault names that are not supplied.\n\n"
        f"Fault metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Relevant extracted report context:\n{report_context}"
    )


def media_prompt(record: dict[str, Any], context: str, count: int) -> str:
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
        "Questions should ask for interpretation of the extracted figure/table. Answers must "
        "be grounded in the caption and nearby report text, and should explain why the visual "
        "matters for seismic or subsurface interpretation.\n\n"
        f"Media metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Extracted visual context:\n{context}"
    )


def chat_input(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return JSON only. Generate grounded seismic QA data."},
        {"role": "user", "content": json_only_prompt(prompt)},
    ]


def normalize_generation(output: Any) -> str:
    if isinstance(output, dict):
        generations = output.get("generations")
        if isinstance(generations, list) and generations:
            first = generations[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return str(first.get("text") or first.get("content") or first)
        return str(output.get("text") or output.get("content") or output)
    if isinstance(output, list) and output:
        return normalize_generation(output[0])
    return str(output)


class DistilabelAnthropicGenerator:
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = AnthropicLLM(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.llm.load()

    def generate_batch(self, prompts: list[str]) -> list[tuple[str, list[dict[str, str]]]]:
        inputs = [chat_input(prompt) for prompt in prompts]
        outputs = self.llm.generate(
            inputs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        parsed = []
        for output in outputs:
            text = normalize_generation(output)
            parsed.append((text, extract_json_array(text)))
        return parsed


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
    id_prefix: str,
) -> dict[str, Any]:
    token = TASK_TOKENS[task]
    record_id = stable_id(
        id_prefix,
        dataset,
        task,
        source.get("slice_index"),
        source.get("doc_id"),
        source.get("element_id"),
        variant,
    )
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
            "generator": "distilabel",
        },
    }


def source_items(
    fault_info: Path,
    interpretation_root: Path,
    report_context: str,
    max_context_chars: int,
    synthetic_per_record: int,
    max_fault_records: int | None,
    max_interpretation_records: int | None,
    fault_only: bool,
    interpretation_only: bool,
) -> list[dict[str, Any]]:
    items = []
    if not interpretation_only:
        fault_records = read_jsonl(fault_info)
        if max_fault_records is not None:
            fault_records = fault_records[:max_fault_records]
        for record in fault_records:
            items.append({
                "kind": "fault",
                "source": record,
                "prompt": fault_prompt(record, report_context, synthetic_per_record),
            })

    if not fault_only:
        media_records = collect_media_records(interpretation_root)
        if max_interpretation_records is not None:
            media_records = media_records[:max_interpretation_records]
        for record in media_records:
            context = media_context(record, max_context_chars)
            items.append({
                "kind": "interpretation",
                "source": record,
                "context": context,
                "prompt": media_prompt(record, context, synthetic_per_record),
            })
    return items


def build_records_from_pairs(
    item: dict[str, Any],
    pairs: list[dict[str, str]],
    val_fraction: float,
    id_prefix: str,
) -> list[dict[str, Any]]:
    source = item["source"]
    output = []
    if item["kind"] == "fault":
        image_path = source["image_path"]
        mask_path = source["mask_path"]
        slice_id = source["slice_index"]
        split = split_name(f"fault:{slice_id}", val_fraction)
        metadata = {
            "survey": source.get("survey"),
            "slice_type": source.get("slice_type"),
            "slice_index": slice_id,
            "has_fault": bool(source.get("has_fault")),
            "fault_names": source.get("fault_names", []),
            "fault_point_count": source.get("fault_point_count", 0),
            "image_shape": source.get("image_shape"),
            "mask_shape": source.get("mask_shape"),
        }
        for variant, pair in enumerate(pairs, start=1):
            output.append(qa_record(
                source=source,
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
                id_prefix=id_prefix,
            ))
        output.append(qa_record(
            source=source,
            task="fault_segmentation",
            dataset="fault_interpretation",
            question="Return the binary fault segmentation target for this seismic section.",
            answer=(
                f"[seg] Use the binary fault mask at {mask_path} as the segmentation target."
                if source.get("has_fault")
                else "[seg] The segmentation target is an empty binary mask because no mapped fault pixels are present."
            ),
            split=split,
            image_path=image_path,
            mask_path=mask_path,
            metadata=metadata,
            variant=999,
            synthetic=False,
            id_prefix=id_prefix,
        ))
        return output

    image_path = source.get("image_path")
    key = f"interpretation:{source.get('doc_id')}:{source.get('element_id')}"
    split = split_name(key, val_fraction)
    metadata = {
        "doc_id": source.get("doc_id"),
        "element_id": source.get("element_id"),
        "media_type": source.get("type"),
        "page": source.get("page"),
        "section_path": source.get("section_path", []),
        "caption": source.get("caption"),
    }
    for variant, pair in enumerate(pairs, start=1):
        output.append(qa_record(
            source=source,
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
            id_prefix=id_prefix,
        ))
    return output


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


def contradiction_flags(record: dict[str, Any]) -> list[str]:
    if record.get("task") != "fault_interpretation":
        return []
    has_fault = record.get("metadata", {}).get("has_fault")
    text = f"{record.get('question', '')} {record.get('answer', '')}".lower()
    flags = []
    positive_patterns = (
        r"(?<!no )(?<!not )fault evidence is present",
        r"(?<!does not )(?<!doesn't )contains? fault",
        r"(?<!no )(?<!not )fault is present",
        r"shows fault",
    )
    negative_phrases = ("no fault", "does not contain", "no evidence of fault", "absence of fault")
    if has_fault is False and any(re.search(pattern, text) for pattern in positive_patterns):
        flags.append("possible_false_positive_language")
    if has_fault is True and any(phrase in text for phrase in negative_phrases):
        flags.append("possible_false_negative_language")
    return flags


def evaluate_records(records: list[dict[str, Any]], output_path: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total": len(records),
        "missing_required_fields": 0,
        "empty_question": 0,
        "empty_answer": 0,
        "missing_image_path": 0,
        "missing_mask_path_for_segmentation": 0,
        "missing_existing_image_file": 0,
        "missing_existing_mask_file": 0,
        "duplicate_ids": 0,
        "task_token_mismatch": 0,
        "possible_fault_contradictions": 0,
        "synthetic_true": 0,
        "synthetic_false": 0,
        "by_dataset": {},
        "by_task": {},
        "by_split": {},
        "examples": {
            "missing_files": [],
            "contradictions": [],
        },
    }
    seen = set()
    for record in records:
        report["by_dataset"][record.get("dataset")] = report["by_dataset"].get(record.get("dataset"), 0) + 1
        report["by_task"][record.get("task")] = report["by_task"].get(record.get("task"), 0) + 1
        report["by_split"][record.get("split")] = report["by_split"].get(record.get("split"), 0) + 1
        report["missing_required_fields"] += bool(REQUIRED_FIELDS - set(record))
        report["empty_question"] += not bool(str(record.get("question", "")).strip())
        report["empty_answer"] += not bool(str(record.get("answer", "")).strip())

        record_id = record.get("id")
        if record_id in seen:
            report["duplicate_ids"] += 1
        seen.add(record_id)

        expected_token = TASK_TOKENS.get(record.get("task"))
        report["task_token_mismatch"] += bool(expected_token and record.get("task_token") != expected_token)

        synthetic = record.get("metadata", {}).get("synthetic")
        report["synthetic_true"] += synthetic is True
        report["synthetic_false"] += synthetic is False

        image_path = record.get("image_path")
        if not image_path:
            report["missing_image_path"] += 1
        elif not Path(image_path).exists():
            report["missing_existing_image_file"] += 1
            if len(report["examples"]["missing_files"]) < 10:
                report["examples"]["missing_files"].append(image_path)

        mask_path = record.get("mask_path")
        if record.get("task") == "fault_segmentation" and not mask_path:
            report["missing_mask_path_for_segmentation"] += 1
        if mask_path and not Path(mask_path).exists():
            report["missing_existing_mask_file"] += 1
            if len(report["examples"]["missing_files"]) < 10:
                report["examples"]["missing_files"].append(mask_path)

        flags = contradiction_flags(record)
        if flags:
            report["possible_fault_contradictions"] += 1
            if len(report["examples"]["contradictions"]) < 10:
                report["examples"]["contradictions"].append({
                    "id": record.get("id"),
                    "flags": flags,
                    "has_fault": record.get("metadata", {}).get("has_fault"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                })

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def generate_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    if load_dotenv is not None:
        load_dotenv(args.env_file)
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required. Pass --api-key or set it in the environment/.env.")

    report_context = collect_report_chunks(args.interpretation_root, args.max_report_context_chars)
    items = source_items(
        fault_info=args.fault_info,
        interpretation_root=args.interpretation_root,
        report_context=report_context,
        max_context_chars=args.max_context_chars,
        synthetic_per_record=args.synthetic_per_record,
        max_fault_records=args.max_fault_records or None,
        max_interpretation_records=args.max_interpretation_records or None,
        fault_only=args.fault_only,
        interpretation_only=args.interpretation_only,
    )
    if not items:
        raise ValueError("No source records found for generation.")

    generator = DistilabelAnthropicGenerator(
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    records = []
    raw_records = []
    for start in tqdm(range(0, len(items), args.batch_size), desc="Distilabel synthetic QA"):
        batch = items[start:start + args.batch_size]
        generated = generator.generate_batch([item["prompt"] for item in batch])
        for item, (raw_text, pairs) in zip(batch, generated):
            raw_records.append({
                "kind": item["kind"],
                "source_id": item["source"].get("slice_index") or item["source"].get("element_id"),
                "raw_generation": raw_text,
                "parsed_pairs": pairs,
            })
            records.extend(build_records_from_pairs(
                item,
                pairs,
                val_fraction=args.val_fraction,
                id_prefix=args.id_prefix,
            ))

    if args.raw_output:
        write_jsonl(args.raw_output, raw_records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate multimodal QA JSONL using Distilabel + Anthropic."
    )
    parser.add_argument("--fault-info", type=Path, default=Path("process_data/fault_detection/data/info.jsonl"))
    parser.add_argument("--interpretation-root", type=Path, default=Path("process_data/interpretation"))
    parser.add_argument("--output-dir", type=Path, default=Path("process_data/multimodal_qa"))
    parser.add_argument("--raw-output", type=Path, default=Path("process_data/multimodal_qa/raw_distilabel_generations.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Evaluate an existing JSONL instead of generating.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fault-only", action="store_true")
    parser.add_argument("--interpretation-only", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-report-context-chars", type=int, default=12000)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--synthetic-per-record", type=int, default=2)
    parser.add_argument("--max-fault-records", type=int, default=0)
    parser.add_argument("--max-interpretation-records", type=int, default=0)
    parser.add_argument("--model", default="claude-sonnet-4-0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--id-prefix", default="distilabel")
    args = parser.parse_args()

    if args.evaluate_only:
        input_path = args.input_jsonl or args.output_dir / "all.jsonl"
        records = read_jsonl(input_path)
        eval_path = args.eval_output or input_path.parent / "evaluation.json"
        report = evaluate_records(records, eval_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.fault_only and args.interpretation_only:
        raise ValueError("Use only one of --fault-only or --interpretation-only.")

    records = generate_dataset(args)
    save_outputs(records, args.output_dir)
    eval_path = args.eval_output or args.output_dir / "evaluation.json"
    report = evaluate_records(records, eval_path)
    print(f"wrote {len(records)} QA records to {args.output_dir}")
    print(f"wrote evaluation report to {eval_path}")
    print(json.dumps({
        "total": report["total"],
        "synthetic_true": report["synthetic_true"],
        "synthetic_false": report["synthetic_false"],
        "by_task": report["by_task"],
        "missing_existing_image_file": report["missing_existing_image_file"],
        "missing_existing_mask_file": report["missing_existing_mask_file"],
        "possible_fault_contradictions": report["possible_fault_contradictions"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
