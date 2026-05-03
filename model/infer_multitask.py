import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPImageProcessor

from multitask import (
    SEGMENTATION_TASKS,
    format_text_sample,
    latest_checkpoint,
    load_checkpoint,
    load_visual,
    read_jsonl,
    user_text,
)


def load_record(path: Path, index: int) -> dict:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No records found in {path}")
    if index < 0 or index >= len(records):
        raise IndexError(f"--record-index {index} is out of range for {len(records)} records")
    return records[index]


def build_prompt(question: str, task_token: str) -> str:
    text = question.strip()
    if not text.startswith(task_token):
        text = f"{task_token} {text}"
    prompt_text, _ = format_text_sample(text, "")
    return prompt_text


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, mask)
        return

    from PIL import Image

    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multitask VLM inference for interpretation, fault QA, or segmentation.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("model/checkpoints/multitask"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--task-token", choices=["[interp]", "[fault]", "[seg]"], default="[interp]")
    parser.add_argument("--record-jsonl", type=Path, default=None)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--mask-output", type=Path, default=Path("outputs/predicted_mask.npy"))
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--print-reference", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_root)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    image_processor = CLIPImageProcessor.from_pretrained(model.vision_name)

    reference_answer = None
    record_task = None
    if args.record_jsonl is not None:
        record = load_record(args.record_jsonl, args.record_index)
        image_path = Path(record["image_path"])
        task_token = record.get("task_token", args.task_token)
        record_task = record.get("task")
        prompt, _ = format_text_sample(user_text(record), "")
        reference_answer = record.get("answer")
    else:
        if args.image is None:
            raise ValueError("Pass --image, or pass --record-jsonl to infer from a dataset record.")
        task_token = args.task_token
        image_path = args.image
        question = args.question or {
            "[interp]": "What does this report figure or table show, and why is it useful for seismic interpretation?",
            "[fault]": "Does this seismic section contain mapped fault evidence? Answer yes or no and briefly justify.",
            "[seg]": "Return the fault segmentation target for this seismic section.",
        }[task_token]
        prompt = build_prompt(question, task_token)

    image = load_visual(image_path)
    pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values.to(device)

    is_segmentation = task_token == "[seg]" or record_task in SEGMENTATION_TASKS
    if is_segmentation:
        with torch.inference_mode():
            logits = model.segment(pixel_values, output_size=image.size[::-1])
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (probs >= args.mask_threshold).astype(np.float32)
        save_mask(mask, args.mask_output)
        print(json.dumps({
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint.get("step"),
            "image": str(image_path),
            "task_token": task_token,
            "mask_output": str(args.mask_output),
            "mask_probability_mean": float(probs.mean()),
            "mask_probability_max": float(probs.max()),
            "reference_answer": reference_answer if args.print_reference else None,
        }, ensure_ascii=False, indent=2))
        return

    tokenized = model.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
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
            input_ids=tokenized.input_ids.to(device),
            attention_mask=tokenized.attention_mask.to(device),
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
