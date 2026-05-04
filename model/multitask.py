import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BitsAndBytesConfig

from model import VLM


IGNORE_INDEX = -100
SEGMENTATION_TASKS = {"fault_segmentation"}
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
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
    return f"User: {prompt}\nAssistant:", f" {answer}"


def load_visual(path: str | Path) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        finite = np.isfinite(array)
        if not finite.all():
            array = np.where(finite, array, 0.0)
        low, high = np.percentile(array, [1, 99])
        if high <= low:
            high = low + 1.0
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
        image = (array * 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    return Image.open(path).convert("RGB")


def load_mask(path: str | Path | None, output_size: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.zeros((1, *output_size), dtype=torch.float32)
    array = np.load(path).astype(np.float32)
    tensor = torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)
    if tensor.shape[-2:] != output_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=output_size,
            mode="nearest",
        ).squeeze(0)
    return tensor


class MultitaskQADataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tasks: set[str] | None = None):
        self.path = Path(jsonl_path)
        records = read_jsonl(self.path)
        self.records = [
            record for record in records
            if tasks is None or record.get("task") in tasks or record.get("task_token") in tasks
        ]
        if not self.records:
            raise ValueError(f"No records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        task = record["task"]
        task_token = record.get("task_token", "[interp]")
        return {
            "id": record["id"],
            "dataset": record.get("dataset"),
            "task": task,
            "task_token": task_token,
            "is_segmentation": task in SEGMENTATION_TASKS or task_token == "[seg]",
            "image_path": record["image_path"],
            "mask_path": record.get("mask_path"),
            "prompt": user_text(record),
            "answer": record["answer"],
            "metadata": record.get("metadata", {}),
        }


class MultitaskCollator:
    def __init__(
        self,
        tokenizer,
        image_processor,
        max_length: int = 1024,
        mask_size: tuple[int, int] = (224, 224),
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.mask_size = mask_size

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
        images = [load_visual(item["image_path"]) for item in batch]
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values

        input_ids = []
        attention_masks = []
        labels = []
        masks = []
        is_segmentation = []
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
            if item["is_segmentation"]:
                sample_labels = [IGNORE_INDEX] * len(sample_ids)
            else:
                sample_labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids + [eos_id])[:self.max_length]

            input_ids.append(torch.tensor(sample_ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(sample_ids), dtype=torch.long))
            labels.append(torch.tensor(sample_labels, dtype=torch.long))
            masks.append(load_mask(item["mask_path"], self.mask_size))
            is_segmentation.append(bool(item["is_segmentation"]))

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_masks,
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
            "masks": torch.stack(masks),
            "is_segmentation": torch.tensor(is_segmentation, dtype=torch.bool),
            "tasks": [item["task"] for item in batch],
            "ids": [item["id"] for item in batch],
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    preds = torch.sigmoid(logits) >= threshold
    targets_bool = targets >= 0.5
    dims = tuple(range(1, preds.ndim))

    intersection = (preds & targets_bool).sum(dim=dims).float()
    union = (preds | targets_bool).sum(dim=dims).float()
    pred_sum = preds.sum(dim=dims).float()
    target_sum = targets_bool.sum(dim=dims).float()

    has_target = target_sum > 0
    dice_per_item = 2 * intersection / (pred_sum + target_sum).clamp_min(1)
    iou_per_item = intersection / union.clamp_min(1)

    positive_dice = dice_per_item[has_target].mean().item() if has_target.any() else 0.0
    positive_iou = iou_per_item[has_target].mean().item() if has_target.any() else 0.0
    return {
        "seg_dice": dice_per_item.mean().item(),
        "seg_iou": iou_per_item.mean().item(),
        "seg_dice_positive": positive_dice,
        "seg_iou_positive": positive_iou,
        "seg_pred_positive_ratio": preds.float().mean().item(),
        "seg_target_positive_ratio": targets_bool.float().mean().item(),
        "seg_positive_mask_ratio": has_target.float().mean().item(),
    }


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_multitask_loss(
    model: VLM,
    batch: dict[str, Any],
    seg_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if (batch["labels"] != IGNORE_INDEX).any():
        text_loss = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
    else:
        text_loss = torch.zeros((), device=batch["pixel_values"].device)

    seg_mask = batch["is_segmentation"]
    if seg_mask.any():
        seg_logits = model.segment(
            batch["pixel_values"][seg_mask],
            output_size=batch["masks"].shape[-2:],
        )
        seg_targets = batch["masks"][seg_mask].to(dtype=seg_logits.dtype)
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, seg_targets) + dice_loss(seg_logits, seg_targets)
        metrics = segmentation_metrics(seg_logits.detach(), seg_targets.detach())
    else:
        seg_loss = torch.zeros((), device=batch["pixel_values"].device)
        metrics = {
            "seg_dice": 0.0,
            "seg_iou": 0.0,
            "seg_dice_positive": 0.0,
            "seg_iou_positive": 0.0,
            "seg_pred_positive_ratio": 0.0,
            "seg_target_positive_ratio": 0.0,
            "seg_positive_mask_ratio": 0.0,
        }

    loss = text_loss_weight * text_loss + seg_loss_weight * seg_loss
    metrics.update({
        "loss": float(loss.detach().cpu()),
        "text_loss": float(text_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
    })
    return loss, metrics


def evaluate(
    model: VLM,
    dataloader: DataLoader,
    device: torch.device,
    seg_loss_weight: float = 1.0,
    text_loss_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "text_loss": 0.0,
        "seg_loss": 0.0,
        "seg_dice": 0.0,
        "seg_iou": 0.0,
        "seg_dice_positive": 0.0,
        "seg_iou_positive": 0.0,
        "seg_pred_positive_ratio": 0.0,
        "seg_target_positive_ratio": 0.0,
        "seg_positive_mask_ratio": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for batch_count, batch in enumerate(dataloader, start=1):
            _, metrics = compute_multitask_loss(
                model,
                to_device(batch, device),
                seg_loss_weight=seg_loss_weight,
                text_loss_weight=text_loss_weight,
            )
            for key in totals:
                totals[key] += metrics[key]
    model.train()
    return {key: value / max(batch_count, 1) for key, value in totals.items()}


def balanced_sampler(dataset: MultitaskQADataset, enabled: bool):
    if not enabled:
        return None
    counts: dict[str, int] = {}
    for record in dataset.records:
        token = record.get("task_token", "")
        counts[token] = counts.get(token, 0) + 1
    weights = [1.0 / counts[record.get("task_token", "")] for record in dataset.records]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def save_checkpoint(
    model: VLM,
    output_dir: Path,
    step: int,
    epoch: int,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "vision_name": model.vision_name,
        "llm_name": model.llm_name,
        "num_query_tokens": model.num_query_tokens,
        "task_tokens": model.task_tokens,
        **metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint_dir / "pytorch_model.pt")
    model.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")


def build_qlora_config() -> BitsAndBytesConfig:
    if not torch.cuda.is_available():
        details = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_available": torch.cuda.is_available(),
        }
        raise RuntimeError(
            "QLoRA requires a CUDA-visible NVIDIA GPU because 4-bit bitsandbytes "
            f"training is not available on CPU. PyTorch CUDA diagnostics: {details}. "
            "If this machine has an NVIDIA GPU, install a CUDA-enabled PyTorch wheel "
            "and verify `nvidia-smi` works in the same shell."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError("QLoRA requires bitsandbytes. Install it on Linux/CUDA with: uv sync --extra qlora") from exc

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


def attach_lora_for_loading(model: VLM, checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("training_mode") != "qlora":
        return
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


def load_runtime_state_dict(model: VLM, checkpoint_state: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if "quant_state" in key or key.endswith((".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")):
            skipped.append(key)
            continue
        if ".base_layer.weight" in key:
            skipped.append(key)
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        loadable[key] = value

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if skipped:
        print(f"checkpoint load: skipped {len(skipped)} incompatible or quantized keys")
    if missing:
        print(f"checkpoint load: missing {len(missing)} model keys after partial load")
    if unexpected:
        print(f"checkpoint load: unexpected {len(unexpected)} keys after partial load")


def move_trainable_vlm_parts_to_device(model: VLM, device: torch.device) -> None:
    model.vision_encoder.to(device)
    model.Qformer.to(device)
    model.visual_projection.to(device)
    model.segmentation_decoder.to(device)
    model.query_tokens.data = model.query_tokens.data.to(device)


def build_model(
    vision_name: str,
    llm_name: str,
    training_mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
    gradient_checkpointing: bool,
    device: torch.device,
) -> VLM:
    quantization_config = build_qlora_config() if training_mode == "qlora" else None
    if training_mode == "qlora" and platform.system() != "Linux":
        print(
            "warning: QLoRA depends on bitsandbytes 4-bit CUDA support. "
            f"Current OS is {platform.system()}; if loading fails, use Linux/CUDA "
            "or run --training-mode frozen."
        )

    model = VLM(
        vision_name=vision_name,
        llm_name=llm_name,
        llm_quantization_config=quantization_config,
        llm_device_map="auto" if training_mode == "qlora" else None,
        freeze_llm=True,
    )
    if training_mode == "qlora":
        apply_qlora(
            model=model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )
        move_trainable_vlm_parts_to_device(model, device)
        model.llm.print_trainable_parameters()
    else:
        model.to(device)
    return model


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
        vision_name=checkpoint.get("vision_name", "facebook/dinov2-base"),
        llm_name=checkpoint.get("llm_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        num_query_tokens=checkpoint.get("num_query_tokens", 32),
        task_tokens=checkpoint.get("task_tokens", ("[interp]", "[fault]", "[seg]")),
    )
    attach_lora_for_loading(model, checkpoint)
    load_runtime_state_dict(model, checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def trainable_parameters(model: torch.nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def parse_lora_targets(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    return max(1, math.ceil(num_batches / grad_accum_steps))
