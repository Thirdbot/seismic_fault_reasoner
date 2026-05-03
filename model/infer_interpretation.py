import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPImageProcessor

from model import VLM

try:
    from .train_interpretation import format_text_sample, read_jsonl, user_text
except ImportError:
    from train_interpretation import format_text_sample, read_jsonl, user_text


def latest_checkpoint(root: Path) -> Path:
    checkpoints = sorted(
        root.glob("checkpoint-step-*/pytorch_model.pt"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-step-*/pytorch_model.pt found in {root}")
    return checkpoints[-1]


def load_checkpoint(path: Path, device: torch.device) -> tuple[VLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VLM(
        vision_name=checkpoint.get("vision_name", "openai/clip-vit-base-patch32"),
        llm_name=checkpoint.get("llm_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        num_query_tokens=checkpoint.get("num_query_tokens", 32),
        task_tokens=checkpoint.get("task_tokens", ("[interp]", "[fault]", "[seg]")),
    )
    if checkpoint.get("training_mode") == "qlora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("Loading a QLoRA checkpoint requires peft. Install with: uv sync --extra qlora") from exc

        lora_config = LoraConfig(
            r=checkpoint["lora_r"],
            lora_alpha=checkpoint["lora_alpha"],
            target_modules=checkpoint["lora_target_modules"],
            lora_dropout=checkpoint["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lora_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_record(path: Path, index: int) -> dict[str, Any]:
    records = [
        record for record in read_jsonl(path)
        if record.get("dataset") == "seismic_interpretation"
    ]
    if not records:
        raise ValueError(f"No seismic_interpretation records found in {path}")
    if index < 0 or index >= len(records):
        raise IndexError(f"--record-index {index} is out of range for {len(records)} records")
    return records[index]


def build_prompt(question: str) -> str:
    text = question.strip()
    if not text.startswith("[interp]"):
        text = f"[interp] {text}"
    prompt_text, _ = format_text_sample(text, "")
    return prompt_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interpretation inference with a trained QFormer VLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("model/checkpoints/interpretation"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--question", default="What does this report figure or table show, and why is it useful for seismic interpretation?")
    parser.add_argument("--record-jsonl", type=Path, default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--print-reference", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_root)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    image_processor = CLIPImageProcessor.from_pretrained(model.vision_name)

    reference_answer = None
    if args.record_jsonl is not None:
        record = load_record(args.record_jsonl, args.record_index)
        image_path = Path(record["image_path"])
        prompt, _ = format_text_sample(user_text(record), "")
        reference_answer = record.get("answer")
    else:
        if args.image is None:
            raise ValueError("Pass --image, or pass --record-jsonl to infer from a dataset record.")
        image_path = args.image
        prompt = build_prompt(args.question)

    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values.to(device)
    tokenized = model.tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    answer = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "image": str(image_path),
        "prompt": prompt,
        "answer": answer,
        "reference_answer": reference_answer if args.print_reference else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
