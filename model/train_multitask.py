import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPImageProcessor, get_linear_schedule_with_warmup

from multitask import (
    MultitaskCollator,
    MultitaskQADataset,
    build_model,
    count_trainable_parameters,
    dice_loss,
    optimizer_steps_per_epoch,
    parse_lora_targets,
    segmentation_metrics,
    trainable_parameters,
)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_losses(
    model,
    batch: dict[str, Any],
    seg_loss_weight: float,
    text_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    has_text_labels = (batch["labels"] != -100).any()
    if has_text_labels:
        text_outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        text_loss = text_outputs.loss
    else:
        text_loss = torch.zeros((), device=batch["pixel_values"].device)

    seg_mask = batch["is_segmentation"]
    if seg_mask.any():
        seg_logits = model.segment(
            batch["pixel_values"][seg_mask],
            output_size=batch["masks"].shape[-2:],
        )
        seg_targets = batch["masks"][seg_mask].to(dtype=seg_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets)
        d_loss = dice_loss(seg_logits, seg_targets)
        seg_loss = bce + d_loss
        metrics = segmentation_metrics(seg_logits.detach(), seg_targets.detach())
    else:
        seg_loss = torch.zeros((), device=batch["pixel_values"].device)
        metrics = {"seg_dice": 0.0, "seg_iou": 0.0}

    total_loss = text_loss_weight * text_loss + seg_loss_weight * seg_loss
    metrics.update({
        "loss": float(total_loss.detach().cpu()),
        "text_loss": float(text_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
    })
    return total_loss, metrics


def evaluate(model, dataloader: DataLoader, device: torch.device, seg_loss_weight: float, text_loss_weight: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "text_loss": 0.0, "seg_loss": 0.0, "seg_dice": 0.0, "seg_iou": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            _, metrics = compute_losses(model, batch, seg_loss_weight, text_loss_weight)
            for key in totals:
                totals[key] += metrics[key]
            batches += 1
    model.train()
    return {key: value / max(batches, 1) for key, value in totals.items()}


def save_checkpoint(model, output_dir: Path, step: int, epoch: int, metadata: dict[str, Any]) -> None:
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


def make_sampler(dataset: MultitaskQADataset, balance_tasks: bool):
    if not balance_tasks:
        return None
    counts: dict[str, int] = {}
    for record in dataset.records:
        token = record.get("task_token", "")
        counts[token] = counts.get(token, 0) + 1
    weights = [1.0 / counts[record.get("task_token", "")] for record in dataset.records]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QFormer VLM on interpretation, fault QA, and segmentation.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("process_data/multimodal_qa/train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("process_data/multimodal_qa/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/checkpoints/multitask"))
    parser.add_argument("--vision-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--llm-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--training-mode", choices=["qlora", "frozen"], default="qlora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--mask-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--balance-tasks", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_targets = parse_lora_targets(args.lora_target_modules)
    model = build_model(
        vision_name=args.vision_name,
        llm_name=args.llm_name,
        training_mode=args.training_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        gradient_checkpointing=args.gradient_checkpointing,
        device=device,
    )
    model.train()

    image_processor = CLIPImageProcessor.from_pretrained(args.vision_name)
    collator = MultitaskCollator(
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
        mask_size=(args.mask_size, args.mask_size),
    )
    train_dataset = MultitaskQADataset(args.train_jsonl)
    val_dataset = MultitaskQADataset(args.val_jsonl)
    sampler = make_sampler(train_dataset, args.balance_tasks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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

    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), args.grad_accum_steps)
    total_update_steps = max(1, steps_per_epoch * args.epochs)
    trainable_count, total_count = count_trainable_parameters(model)
    print(
        "training config: "
        f"mode={args.training_mode}, epochs={args.epochs}, train_batches={len(train_loader)}, "
        f"grad_accum_steps={args.grad_accum_steps}, max_steps={args.max_steps}, "
        f"planned_optimizer_steps={total_update_steps}, balance_tasks={args.balance_tasks}"
    )
    print(f"trainable parameters: {trainable_count:,} / {total_count:,}")

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
    checkpoint_metadata = {
        "training_mode": args.training_mode,
        "lora_r": args.lora_r if args.training_mode == "qlora" else None,
        "lora_alpha": args.lora_alpha if args.training_mode == "qlora" else None,
        "lora_dropout": args.lora_dropout if args.training_mode == "qlora" else None,
        "lora_target_modules": lora_targets if args.training_mode == "qlora" else None,
        "seg_loss_weight": args.seg_loss_weight,
        "text_loss_weight": args.text_loss_weight,
        "mask_size": args.mask_size,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = batch_to_device(batch, device)
            loss, metrics = compute_losses(model, batch, args.seg_loss_weight, args.text_loss_weight)
            (loss / args.grad_accum_steps).backward()

            if micro_step % args.grad_accum_steps != 0:
                progress.set_postfix(loss=metrics["loss"], text=metrics["text_loss"], seg=metrics["seg_loss"])
                continue

            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.set_postfix(loss=metrics["loss"], text=metrics["text_loss"], seg=metrics["seg_loss"])

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
                print(f"step={global_step} val={val_metrics}")
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)
            if args.max_steps > 0 and global_step >= args.max_steps:
                val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
                print(f"max_steps reached at step={global_step}; val={val_metrics}")
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)
                return

        if len(train_loader) % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
        print(f"epoch={epoch} val={val_metrics}")
        save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata)


if __name__ == "__main__":
    main()
