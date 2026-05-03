import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig, CLIPImageProcessor, get_linear_schedule_with_warmup

from model import VLM


IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def user_text(record: dict[str, Any]) -> str:
    for message in record.get("messages", []):
        if message.get("role") != "user":
            continue
        for item in message.get("content", []):
            if item.get("type") == "text":
                return item["text"]

    task_token = record.get("task_token", "[interp]")
    return f"{task_token} {record['question']}"


def format_text_sample(prompt: str, answer: str) -> tuple[str, str]:
    prompt_text = f"User: {prompt}\nAssistant:"
    answer_text = f" {answer}"
    return prompt_text, answer_text


class InterpretationQADataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.records = [
            record for record in read_jsonl(self.path)
            if record.get("dataset") == "seismic_interpretation"
        ]
        if not self.records:
            raise ValueError(f"No seismic_interpretation records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "id": record["id"],
            "image_path": record["image_path"],
            "prompt": user_text(record),
            "answer": record["answer"],
            "task_token": record.get("task_token", "[interp]"),
        }


class InterpretationCollator:
    def __init__(
        self,
        tokenizer,
        image_processor,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values

        input_ids = []
        attention_masks = []
        labels = []
        for item in batch:
            prompt_text, answer_text = format_text_sample(item["prompt"], item["answer"])
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            ).input_ids
            answer_ids = self.tokenizer(
                answer_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, self.max_length - len(prompt_ids) - 1),
            ).input_ids
            eos_id = self.tokenizer.eos_token_id
            sample_ids = (prompt_ids + answer_ids + [eos_id])[:self.max_length]
            sample_labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids + [eos_id]
            sample_labels = sample_labels[:self.max_length]

            input_ids.append(torch.tensor(sample_ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(sample_ids), dtype=torch.long))
            labels.append(torch.tensor(sample_labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def trainable_parameters(model: torch.nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_qlora_config() -> BitsAndBytesConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU because 4-bit bitsandbytes training is not available on CPU.")

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "QLoRA requires bitsandbytes. Install it on Linux/CUDA with: "
            "uv sync --extra qlora"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )


def apply_qlora(
    model: VLM,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: bool,
) -> None:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("QLoRA requires peft. Install dependencies with: uv sync --extra qlora") from exc

    model.llm = prepare_model_for_kbit_training(
        model.llm,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_config)


def move_trainable_vlm_parts_to_device(model: VLM, device: torch.device) -> None:
    model.vision_encoder.to(device)
    model.Qformer.to(device)
    model.visual_projection.to(device)
    model.query_tokens.data = model.query_tokens.data.to(device)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def save_checkpoint(model: VLM, output_dir: Path, step: int, epoch: int, metadata: dict[str, Any]) -> None:
    checkpoint_dir = output_dir / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "vision_name": model.vision_name,
            "llm_name": model.llm_name,
            "num_query_tokens": model.num_query_tokens,
            "task_tokens": model.task_tokens,
            **metadata,
        },
        checkpoint_dir / "pytorch_model.pt",
    )
    model.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")


def evaluate(model: VLM, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.detach().cpu())
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QFormer VLM on interpretation QA records.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("process_data/multimodal_qa/train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("process_data/multimodal_qa/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/checkpoints/interpretation"))
    parser.add_argument("--vision-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--llm-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--training-mode", choices=["qlora", "frozen"], default="qlora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Qwen module names to receive LoRA adapters.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps. Use 0 to train for all epochs.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization_config = build_qlora_config() if args.training_mode == "qlora" else None
    if args.training_mode == "qlora" and platform.system() != "Linux":
        print(
            "warning: QLoRA depends on bitsandbytes 4-bit CUDA support. "
            f"Current OS is {platform.system()}; if loading fails, use Linux/CUDA "
            "or run --training-mode frozen."
        )

    model = VLM(
        vision_name=args.vision_name,
        llm_name=args.llm_name,
        llm_quantization_config=quantization_config,
        llm_device_map="auto" if args.training_mode == "qlora" else None,
        freeze_llm=True,
    )
    if args.training_mode == "qlora":
        apply_qlora(
            model=model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
            gradient_checkpointing=args.gradient_checkpointing,
        )
        move_trainable_vlm_parts_to_device(model, device)
        model.llm.print_trainable_parameters()
    else:
        model.to(device)
    model.train()

    image_processor = CLIPImageProcessor.from_pretrained(args.vision_name)
    collator = InterpretationCollator(
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )

    train_dataset = InterpretationQADataset(args.train_jsonl)
    val_dataset = InterpretationQADataset(args.val_jsonl)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    total_update_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.grad_accum_steps))
    print(
        "training config: "
        f"mode={args.training_mode}, epochs={args.epochs}, train_batches={len(train_loader)}, "
        f"grad_accum_steps={args.grad_accum_steps}, max_steps={args.max_steps}, "
        f"planned_optimizer_steps={total_update_steps}"
    )
    trainable_count, total_count = count_trainable_parameters(model)
    print(f"trainable parameters: {trainable_count:,} / {total_count:,}")
    if args.max_steps > 0:
        print(f"max_steps is active; training will stop after {args.max_steps} optimizer steps.")
    checkpoint_metadata = {
        "training_mode": args.training_mode,
        "lora_r": args.lora_r if args.training_mode == "qlora" else None,
        "lora_alpha": args.lora_alpha if args.training_mode == "qlora" else None,
        "lora_dropout": args.lora_dropout if args.training_mode == "qlora" else None,
        "lora_target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        if args.training_mode == "qlora"
        else None,
    }

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_update_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if micro_step % args.grad_accum_steps != 0:
                progress.set_postfix(loss=float(loss.detach().cpu()) * args.grad_accum_steps)
                continue

            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            progress.set_postfix(loss=float(loss.detach().cpu()) * args.grad_accum_steps)

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step={global_step} val_loss={val_loss:.4f}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)

            if args.max_steps > 0 and global_step >= args.max_steps:
                val_loss = evaluate(model, val_loader, device)
                print(f"max_steps reached at step={global_step}; val_loss={val_loss:.4f}")
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)
                return

        if len(train_loader) % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_loss={val_loss:.4f}")
        save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)


if __name__ == "__main__":
    main()
