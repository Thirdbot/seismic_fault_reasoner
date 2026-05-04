import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, get_linear_schedule_with_warmup

from multitask import (
    DEFAULT_LORA_TARGETS,
    MultitaskCollator,
    MultitaskQADataset,
    balanced_sampler,
    build_model,
    compute_multitask_loss,
    count_trainable_parameters,
    evaluate,
    latest_checkpoint,
    load_runtime_state_dict,
    parse_lora_targets,
    save_checkpoint,
    to_device,
    trainable_parameters,
)


def log_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QFormer VLM on interpretation, fault QA, and segmentation.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("process_data/multimodal_qa/train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("process_data/multimodal_qa/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/checkpoints/multitask"))
    parser.add_argument("--vision-name", default="facebook/dinov2-base")
    parser.add_argument("--llm-name", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--training-mode", choices=["qlora", "frozen"], default="qlora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--mask-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--balance-tasks", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("results/multitask"))
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --output-dir.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume from a specific pytorch_model.pt checkpoint.")
    parser.add_argument("--reset-metrics", action="store_true", help="Overwrite metrics.jsonl instead of appending on resume.")
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

    image_processor = AutoImageProcessor.from_pretrained(args.vision_name)
    collator = MultitaskCollator(
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
        mask_size=(args.mask_size, args.mask_size),
    )
    train_dataset = MultitaskQADataset(args.train_jsonl)
    val_dataset = MultitaskQADataset(args.val_jsonl)
    sampler = balanced_sampler(train_dataset, args.balance_tasks)
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

    total_update_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.grad_accum_steps))
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
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "metrics.jsonl"
    global_step = 0
    start_epoch = 1
    resume_path = args.resume_checkpoint
    if args.resume and resume_path is None:
        resume_path = latest_checkpoint(args.output_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        load_runtime_state_dict(model, checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("resume warning: checkpoint has no optimizer_state_dict; optimizer starts fresh")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print("resume warning: checkpoint has no scheduler_state_dict; scheduler starts fresh")
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed from {resume_path} at epoch={checkpoint.get('epoch')} step={global_step}")

    if not resume_path or args.reset_metrics or not metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
        log_metrics(metrics_path, {
            "type": "config",
            "training_mode": args.training_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "balance_tasks": args.balance_tasks,
            "train_records": len(train_dataset),
            "val_records": len(val_dataset),
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
        })
    else:
        log_metrics(metrics_path, {
            "type": "resume",
            "checkpoint": str(resume_path),
            "start_epoch": start_epoch,
            "step": global_step,
        })

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            loss, metrics = compute_multitask_loss(model, batch, args.seg_loss_weight, args.text_loss_weight)
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
            log_metrics(metrics_path, {
                "type": "train",
                "epoch": epoch,
                "step": global_step,
                **metrics,
            })

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
                print(f"step={global_step} val={val_metrics}")
                log_metrics(metrics_path, {
                    "type": "val",
                    "epoch": epoch,
                    "step": global_step,
                    **val_metrics,
                })
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata, optimizer, scheduler)
            if args.max_steps > 0 and global_step >= args.max_steps:
                val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
                print(f"max_steps reached at step={global_step}; val={val_metrics}")
                log_metrics(metrics_path, {
                    "type": "val",
                    "epoch": epoch,
                    "step": global_step,
                    **val_metrics,
                })
                save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata, optimizer, scheduler)
                return

        if len(train_loader) % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_metrics = evaluate(model, val_loader, device, args.seg_loss_weight, args.text_loss_weight)
        print(f"epoch={epoch} val={val_metrics}")
        log_metrics(metrics_path, {
            "type": "val",
            "epoch": epoch,
            "step": global_step,
            **val_metrics,
        })
        save_checkpoint(model, args.output_dir, global_step, epoch, checkpoint_metadata, optimizer, scheduler)


if __name__ == "__main__":
    main()
