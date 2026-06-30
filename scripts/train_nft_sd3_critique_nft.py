"""Critique-NFT image-main training path following DiffusionNFT script layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIFFUSION_NFT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = DIFFUSION_NFT_ROOT / "dataset"
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from critique_nft import BASE_CONDITION, bad_condition, fix_condition

FAILURE_KEYS = (
    "alignment_missing_object",
    "alignment_spatial_reversal",
    "ocr_misspelled_text",
    "ocr_unreadable_text",
    "quality_blur_noise",
    "quality_blocky_texture",
    "aesthetic_poor_lighting",
    "aesthetic_visual_clutter",
)

FAILURE_TEXT: dict[str, str] = {
    "alignment_missing_object": "the requested subject is missing or replaced by a wrong object",
    "alignment_spatial_reversal": "the left-right or above-below spatial relation is reversed",
    "ocr_misspelled_text": "the requested text is misspelled",
    "ocr_unreadable_text": "the text is warped or unreadable",
    "quality_blur_noise": "local blur and noise obscure important details",
    "quality_blocky_texture": "blocky artifacts and corrupted textures are visible",
    "aesthetic_poor_lighting": "lighting and contrast make the scene visually weak",
    "aesthetic_visual_clutter": "the composition is cluttered and distracts from the subject",
}

FAILURE_PARAPHRASE: dict[str, str] = {
    "alignment_missing_object": "an object named in the prompt is absent from the image",
    "alignment_spatial_reversal": "the requested spatial ordering is flipped",
    "ocr_misspelled_text": "the rendered word has incorrect letters",
    "ocr_unreadable_text": "the lettering cannot be read clearly",
    "quality_blur_noise": "detail is lost because parts of the image are fuzzy and noisy",
    "quality_blocky_texture": "compression-like blocks and damaged texture patches stand out",
    "aesthetic_poor_lighting": "the scene needs more pleasing light and tonal balance",
    "aesthetic_visual_clutter": "too many distracting elements weaken the composition",
}

FAILURE_RUBRIC: dict[str, str] = {
    "alignment_missing_object": "alignment",
    "alignment_spatial_reversal": "count_color_spatial",
    "ocr_misspelled_text": "ocr",
    "ocr_unreadable_text": "ocr",
    "quality_blur_noise": "piqe_like_quality",
    "quality_blocky_texture": "piqe_like_quality",
    "aesthetic_poor_lighting": "aesthetic",
    "aesthetic_visual_clutter": "human_preference",
}

RUBRICS = (
    "alignment",
    "count_color_spatial",
    "ocr",
    "piqe_like_quality",
    "aesthetic",
    "human_preference",
)

CONDITION_ROLES = ("base", "bad", "fix", "scalar")

SCALAR_CONDITION = (
    "multi-reward scalar mixture over prompt faithfulness, OCR, perceptual quality, "
    "aesthetic quality, and human preference"
)

GENERIC_CRITIQUE = "general prompt faithfulness and image quality should be improved"


@dataclass(frozen=True)
class PromptRecord:
    """One prompt-manifest entry with required 01 metadata and synthetic labels."""

    prompt_id: str
    prompt: str
    prompt_hash: str
    source: str
    category: str
    split: str
    used_for_critique_collection: bool
    used_for_reward_model_training: bool
    used_for_hyperparameter_tuning: bool
    official_benchmark_prompt: bool
    used_for_human_eval: bool
    rubric: str
    failure_key: str
    failure_text: str


@dataclass(frozen=True)
class ImageMainConfig:
    """Configuration for the group-01 debug readiness run."""

    phase: str
    run_name: str
    manifest_root: Path
    artifact_root: Path
    checkpoint_root: Path
    log_path: Path
    dataset_root: Path
    seed: int
    train_count: int
    val_count: int
    test_count: int
    ood_count: int
    max_steps: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    device: str
    wandb_project: str
    wandb_mode: str
    allow_disabled_wandb: bool = False


class ImageMainToyDataset(Dataset[dict[str, Any]]):
    """Manifest-backed tiny image-main dataset for smoke and debug runs."""

    def __init__(self, records: Sequence[PromptRecord]) -> None:
        if not records:
            raise ValueError("records must be non-empty")
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prompt_features = prompt_feature_vector(record)
        target = failure_vector(record.failure_key)
        return {
            "record": record,
            "prompt_features": prompt_features,
            "target": target,
            "fix_condition": condition_feature_vector(
                fix_condition(record.failure_text)
            ),
            "bad_condition": condition_feature_vector(
                bad_condition(record.failure_text)
            ),
            "base_condition": condition_feature_vector(BASE_CONDITION),
            "scalar_condition": condition_feature_vector(SCALAR_CONDITION),
        }


class TinyConditionedFlow(nn.Module):
    """Small condition-aware velocity predictor used for 01 readiness checks."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, prompt_features: torch.Tensor, condition_features: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([prompt_features, condition_features], dim=-1))


def stable_hash(text: str, length: int = 16) -> str:
    """Return a stable short SHA-256 hash."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Return the current source commit, accepting rsync-provided markers."""

    if os.environ.get("SOURCE_GIT_COMMIT"):
        return str(os.environ["SOURCE_GIT_COMMIT"])
    try:
        return subprocess.check_output(  # nosec B603,B607
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def select_device(requested: str) -> torch.device:
    """Resolve a requested or automatic torch device."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a small JSONL file."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_records(path: Path, records: Sequence[PromptRecord]) -> None:
    """Write prompt records as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_records(path: Path) -> list[PromptRecord]:
    """Load prompt records from JSONL."""

    return [PromptRecord(**payload) for payload in read_jsonl(path)]


def generated_prompt(failure_key: str, index: int) -> str:
    """Create a deterministic generated prompt for a failure type."""

    colors = ("red", "blue", "green", "yellow", "white", "black", "purple", "orange")
    objects = ("cube", "sphere", "mug", "chair", "bicycle", "robot", "lamp", "vase")
    scenes = ("studio", "kitchen", "garden", "museum", "library", "workshop")
    words = ("CRITIQUE", "ORBIT", "VECTOR", "FLOW", "SIGNAL", "CANVAS")
    color_a = colors[index % len(colors)]
    color_b = colors[(index * 3 + 1) % len(colors)]
    obj_a = objects[(index * 5 + 2) % len(objects)]
    obj_b = objects[(index * 7 + 3) % len(objects)]
    scene = scenes[index % len(scenes)]
    word = words[index % len(words)]
    suffix = f", catalog study {index:04d}"
    if failure_key == "alignment_missing_object":
        return f"a {color_a} {obj_a} next to a {color_b} {obj_b} in a clean {scene}{suffix}"
    if failure_key == "alignment_spatial_reversal":
        return f"a {color_a} {obj_a} to the left of a {color_b} {obj_b} in a {scene}{suffix}"
    if failure_key == "ocr_misspelled_text":
        return f'a storefront sign that clearly reads "{word}" above a {color_a} door{suffix}'
    if failure_key == "ocr_unreadable_text":
        return f'a poster with crisp large text "{word}" on a wall in a {scene}{suffix}'
    if failure_key == "quality_blur_noise":
        return f"a sharp detailed macro photo of a {color_a} {obj_a} on textured fabric{suffix}"
    if failure_key == "quality_blocky_texture":
        return f"a high-resolution product photo of a {color_a} {obj_a} with clean texture{suffix}"
    if failure_key == "aesthetic_poor_lighting":
        return f"a cinematic portrait of a {color_a} {obj_a} with soft balanced lighting{suffix}"
    if failure_key == "aesthetic_visual_clutter":
        return f"a minimal composition featuring one {color_a} {obj_a} centered in a {scene}{suffix}"
    raise KeyError(f"unknown failure key: {failure_key}")


def make_record(
    split: str,
    index: int,
    prompt: str,
    source: str,
    failure_key: str,
    official_benchmark_prompt: bool,
) -> PromptRecord:
    """Build one manifest record."""

    prompt_id = f"{split}-{index:04d}"
    rubric = FAILURE_RUBRIC[failure_key]
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        prompt_hash=stable_hash(prompt),
        source=source,
        category=rubric,
        split=split,
        used_for_critique_collection=split == "train",
        used_for_reward_model_training=False,
        used_for_hyperparameter_tuning=split == "val",
        official_benchmark_prompt=official_benchmark_prompt,
        used_for_human_eval=split in {"test_id", "test_ood"},
        rubric=rubric,
        failure_key=failure_key,
        failure_text=FAILURE_TEXT[failure_key],
    )


def read_text_prompts(path: Path, limit: int) -> list[str]:
    """Read prompts from a text file if it exists."""

    if not path.exists() or limit <= 0:
        return []
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                prompts.append(text)
            if len(prompts) >= limit:
                break
    return prompts


def read_geneval_prompts(path: Path, limit: int) -> list[str]:
    """Read prompts from a GenEval metadata file if it exists."""

    if not path.exists() or limit <= 0:
        return []
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("prompt"):
                prompts.append(str(payload["prompt"]))
            if len(prompts) >= limit:
                break
    return prompts


def official_ood_prompts(dataset_root: Path, count: int) -> list[tuple[str, str, str]]:
    """Return held-out official prompts from the DiffusionNFT prompt files."""

    if count <= 0:
        return []
    per_source = max(1, math.ceil(count / 3))
    collected: list[tuple[str, str, str]] = []
    for prompt in read_geneval_prompts(
        dataset_root / "geneval" / "test_metadata.jsonl", per_source
    ):
        collected.append(
            ("diffusionnft_geneval_test", prompt, "alignment_missing_object")
        )
    for prompt in read_text_prompts(dataset_root / "ocr" / "test.txt", per_source):
        collected.append(("diffusionnft_ocr_test", prompt, "ocr_misspelled_text"))
    for prompt in read_text_prompts(
        dataset_root / "drawbench" / "test.txt", per_source
    ):
        collected.append(
            ("diffusionnft_drawbench_test", prompt, "aesthetic_visual_clutter")
        )
    return collected[:count]


def generated_records(split: str, count: int, offset: int = 0) -> list[PromptRecord]:
    """Create deterministic non-official prompt records."""

    records: list[PromptRecord] = []
    for index in range(count):
        failure_key = FAILURE_KEYS[(index + offset) % len(FAILURE_KEYS)]
        prompt = generated_prompt(failure_key, index + offset)
        records.append(
            make_record(
                split=split,
                index=index,
                prompt=prompt,
                source="generated_template_disjoint_from_official_tests",
                failure_key=failure_key,
                official_benchmark_prompt=False,
            )
        )
    return records


def create_prompt_manifests(cfg: ImageMainConfig) -> dict[str, Path]:
    """Create the four prompt manifests required by Experiment 01."""

    train_records = generated_records("train", cfg.train_count, offset=0)
    val_records = generated_records("val", cfg.val_count, offset=10_000)
    test_id_records = generated_records("test_id", cfg.test_count, offset=20_000)
    official_prompts = official_ood_prompts(cfg.dataset_root, cfg.ood_count)
    if len(official_prompts) < cfg.ood_count:
        missing = cfg.ood_count - len(official_prompts)
        generated_ood = generated_records("test_ood", missing, offset=30_000)
        official_records = [
            make_record(
                split="test_ood",
                index=index,
                prompt=prompt,
                source=source,
                failure_key=failure_key,
                official_benchmark_prompt=True,
            )
            for index, (source, prompt, failure_key) in enumerate(official_prompts)
        ]
        test_ood_records = official_records + generated_ood
    else:
        test_ood_records = [
            make_record(
                split="test_ood",
                index=index,
                prompt=prompt,
                source=source,
                failure_key=failure_key,
                official_benchmark_prompt=True,
            )
            for index, (source, prompt, failure_key) in enumerate(official_prompts)
        ]

    paths = {
        "train": cfg.manifest_root / "train_prompt_manifest.jsonl",
        "val": cfg.manifest_root / "val_prompt_manifest.jsonl",
        "test_id": cfg.manifest_root / "test_id_manifest.jsonl",
        "test_ood": cfg.manifest_root / "test_ood_manifest.jsonl",
    }
    write_records(paths["train"], train_records)
    write_records(paths["val"], val_records)
    write_records(paths["test_id"], test_id_records)
    write_records(paths["test_ood"], test_ood_records)
    return paths


def validate_manifest_records(
    records: Sequence[PromptRecord], expected_split: str
) -> None:
    """Validate required manifest fields and split semantics."""

    if not records:
        raise ValueError(f"{expected_split} manifest is empty")
    prompt_hashes: set[str] = set()
    for record in records:
        if record.split != expected_split:
            raise ValueError(f"expected split {expected_split}, found {record.split}")
        if record.prompt_hash != stable_hash(record.prompt):
            raise ValueError(f"bad prompt hash for {record.prompt_id}")
        if not record.prompt_id or not record.prompt or not record.source:
            raise ValueError(f"missing required field in {record.prompt_id}")
        prompt_hashes.add(record.prompt_hash)
    if len(prompt_hashes) != len(records):
        raise ValueError(f"{expected_split} manifest contains duplicate prompts")


def load_prompt_manifests(paths: dict[str, Path]) -> dict[str, list[PromptRecord]]:
    """Load and validate the four prompt manifests."""

    manifests = {split: load_records(path) for split, path in paths.items()}
    for split, records in manifests.items():
        validate_manifest_records(records, split)
    train_hashes = {record.prompt_hash for record in manifests["train"]}
    heldout_hashes = {
        record.prompt_hash
        for split in ("val", "test_id", "test_ood")
        for record in manifests[split]
    }
    overlap = train_hashes & heldout_hashes
    if overlap:
        raise ValueError(
            f"train prompts overlap held-out prompts: {sorted(overlap)[:3]}"
        )
    for record in manifests["train"]:
        if record.official_benchmark_prompt:
            raise ValueError("official benchmark prompt leaked into train manifest")
    return manifests


def failure_vector(failure_key: str) -> torch.Tensor:
    """Return the synthetic correction vector for a failure key."""

    vector = torch.zeros(len(FAILURE_KEYS), dtype=torch.float32)
    vector[FAILURE_KEYS.index(failure_key)] = 1.0
    return vector


def hashed_float_features(text: str, dims: int) -> list[float]:
    """Build deterministic small prompt features from a hash."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dims):
        byte = digest[index % len(digest)]
        values.append((float(byte) / 127.5) - 1.0)
    return values


def prompt_feature_vector(record: PromptRecord) -> torch.Tensor:
    """Preprocess a prompt record into fixed features."""

    rubric_features = [1.0 if record.rubric == rubric else 0.0 for rubric in RUBRICS]
    prompt_features = hashed_float_features(record.prompt, 8)
    source_feature = [1.0 if record.official_benchmark_prompt else 0.0]
    return torch.tensor(
        rubric_features + prompt_features + source_feature, dtype=torch.float32
    )


def role_from_condition(text: str) -> str:
    """Infer the condition role from its text."""

    lower = text.lower()
    if "multi-reward scalar" in lower or "scalar mixture" in lower:
        return "scalar"
    if "low-quality generation exhibiting" in lower:
        return "bad"
    if "corrected generation" in lower or "addresses these failures" in lower:
        return "fix"
    return "base"


def text_failure_vector(text: str) -> torch.Tensor:
    """Extract a synthetic failure vector from scalar-free critique text."""

    lower = text.lower()
    vector = torch.zeros(len(FAILURE_KEYS), dtype=torch.float32)
    for index, key in enumerate(FAILURE_KEYS):
        direct = FAILURE_TEXT[key].lower()
        paraphrase = FAILURE_PARAPHRASE[key].lower()
        if direct in lower or paraphrase in lower:
            vector[index] = 1.0

    keyword_hits = {
        "alignment_missing_object": ("missing", "absent", "wrong object", "replaced"),
        "alignment_spatial_reversal": (
            "left-right",
            "above-below",
            "spatial",
            "flipped",
        ),
        "ocr_misspelled_text": ("misspelled", "incorrect letters", "wrong letters"),
        "ocr_unreadable_text": ("unreadable", "cannot be read", "warped"),
        "quality_blur_noise": ("blur", "noise", "fuzzy"),
        "quality_blocky_texture": ("blocky", "corrupted texture", "damaged texture"),
        "aesthetic_poor_lighting": ("lighting", "contrast", "tonal balance"),
        "aesthetic_visual_clutter": ("clutter", "distracting", "composition"),
    }
    for index, key in enumerate(FAILURE_KEYS):
        if any(keyword in lower for keyword in keyword_hits[key]):
            vector[index] = 1.0

    if vector.sum() == 0 and any(
        term in lower for term in ("quality", "faithfulness", "improved")
    ):
        vector += 0.25
    if "multi-reward scalar" in lower or "scalar mixture" in lower:
        vector += 1.0 / len(FAILURE_KEYS)
    return vector.clamp(max=1.0)


def condition_feature_vector(text: str) -> torch.Tensor:
    """Preprocess condition text into synthetic role and critique features."""

    role = role_from_condition(text)
    vector = text_failure_vector(text)
    if role == "bad":
        vector = -vector
    elif role == "base":
        vector = torch.zeros_like(vector)
    role_features = [1.0 if role == candidate else 0.0 for candidate in CONDITION_ROLES]
    return torch.cat([vector, torch.tensor(role_features, dtype=torch.float32)])


def collate_batch(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate dataset examples for training/evaluation."""

    return {
        "records": [example["record"] for example in examples],
        "prompt_features": torch.stack(
            [example["prompt_features"] for example in examples]
        ),
        "target": torch.stack([example["target"] for example in examples]),
        "fix_condition": torch.stack(
            [example["fix_condition"] for example in examples]
        ),
        "bad_condition": torch.stack(
            [example["bad_condition"] for example in examples]
        ),
        "base_condition": torch.stack(
            [example["base_condition"] for example in examples]
        ),
        "scalar_condition": torch.stack(
            [example["scalar_condition"] for example in examples]
        ),
    }


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute the L2 norm over available parameter gradients."""

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().cpu())
    return float(total**0.5)


def cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return row-wise cosine similarity with zero protection."""

    numerator = (prediction * target).sum(dim=-1)
    denominator = prediction.norm(dim=-1).clamp_min(1e-8) * target.norm(
        dim=-1
    ).clamp_min(1e-8)
    return numerator / denominator


def model_dimensions() -> tuple[int, int, int]:
    """Return prompt, condition, and target dimensionality."""

    prompt_dim = len(RUBRICS) + 8 + 1
    condition_dim = len(FAILURE_KEYS) + len(CONDITION_ROLES)
    output_dim = len(FAILURE_KEYS)
    return prompt_dim, condition_dim, output_dim


def make_model(hidden_dim: int, device: torch.device) -> TinyConditionedFlow:
    """Construct a tiny same-substrate model."""

    prompt_dim, condition_dim, output_dim = model_dimensions()
    return TinyConditionedFlow(
        input_dim=prompt_dim + condition_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    ).to(device)


def cnft_loss(
    model: TinyConditionedFlow,
    batch: dict[str, Any],
    device: torch.device,
    include_bad_branch: bool = True,
    include_delta: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the tiny Critique-NFT loss family."""

    prompt = batch["prompt_features"].to(device)
    target = batch["target"].to(device)
    fix_features = batch["fix_condition"].to(device)
    bad_features = batch["bad_condition"].to(device)
    base_features = batch["base_condition"].to(device)
    fix_output = model(prompt, fix_features)
    base_output = model(prompt, base_features)
    loss_cnft = torch.nn.functional.mse_loss(fix_output, target)
    loss_cond = torch.nn.functional.mse_loss(base_output, torch.zeros_like(target))
    total = loss_cnft + 0.15 * loss_cond
    loss_bad = torch.zeros((), device=device)
    loss_delta = torch.zeros((), device=device)
    if include_bad_branch or include_delta:
        bad_output = model(prompt, bad_features)
        if include_bad_branch:
            loss_bad = torch.nn.functional.mse_loss(bad_output, -target)
            total = total + loss_bad
        if include_delta:
            loss_delta = torch.nn.functional.mse_loss(
                fix_output - bad_output, 2.0 * target
            )
            total = total + 0.25 * loss_delta
    metrics = {
        "loss/cnft": float(loss_cnft.detach().cpu()),
        "loss/cond": float(loss_cond.detach().cpu()),
        "loss/bad": float(loss_bad.detach().cpu()),
        "loss/delta": float(loss_delta.detach().cpu()),
    }
    return total, metrics


def scalar_loss(
    model: TinyConditionedFlow,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the tiny multi-reward scalar DiffusionNFT surrogate loss."""

    prompt = batch["prompt_features"].to(device)
    target = batch["target"].to(device)
    scalar_features = batch["scalar_condition"].to(device)
    output = model(prompt, scalar_features)
    loss = torch.nn.functional.mse_loss(output, target)
    return loss, {"loss/scalar": float(loss.detach().cpu())}


def positive_only_loss(
    model: TinyConditionedFlow,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the tiny quality-matched positive-only target loss."""

    prompt = batch["prompt_features"].to(device)
    target = batch["target"].to(device)
    base_features = batch["base_condition"].to(device)
    output = model(prompt, base_features)
    loss = torch.nn.functional.mse_loss(output, target)
    return loss, {"loss/positive_only": float(loss.detach().cpu())}


def condition_for_variant(
    record: PromptRecord,
    variant: str,
    records: Sequence[PromptRecord],
    index: int,
) -> tuple[str, str]:
    """Return condition text and critique text for a semantic-control variant."""

    if variant == "correct":
        critique = record.failure_text
    elif variant == "paraphrase":
        critique = FAILURE_PARAPHRASE[record.failure_key]
    elif variant == "same_rubric_wrong_instance":
        candidates = [
            key
            for key in FAILURE_KEYS
            if FAILURE_RUBRIC[key] == record.rubric and key != record.failure_key
        ]
        if not candidates:
            candidates = [key for key in FAILURE_KEYS if key != record.failure_key]
        critique = FAILURE_TEXT[candidates[index % len(candidates)]]
    elif variant == "prompt_only_revision":
        critique = "improve image quality and prompt faithfulness"
    elif variant == "generic":
        critique = GENERIC_CRITIQUE
    elif variant == "random_shuffled":
        other = records[(index * 7 + 3) % len(records)]
        if other.failure_key == record.failure_key:
            other = records[(index * 7 + 4) % len(records)]
        critique = other.failure_text
    elif variant == "wrong_rubric":
        candidates = [
            key
            for key in FAILURE_KEYS
            if FAILURE_RUBRIC[key] != record.rubric and key != record.failure_key
        ]
        critique = FAILURE_TEXT[candidates[index % len(candidates)]]
    else:
        raise ValueError(f"unknown semantic-control variant: {variant}")
    return fix_condition(critique), critique


def evaluate_outputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    records: Sequence[PromptRecord],
) -> dict[str, float]:
    """Compute synthetic image-main metrics for a set of predictions."""

    mse_rows = torch.nn.functional.mse_loss(
        predictions, targets, reduction="none"
    ).mean(dim=-1)
    cosine_rows = cosine_similarity(predictions, targets)
    draft = -targets
    draft_mse = torch.nn.functional.mse_loss(draft, targets, reduction="none").mean(
        dim=-1
    )
    revision_win = (mse_rows < draft_mse).float()
    metrics: dict[str, float] = {
        "mse": float(mse_rows.mean().cpu()),
        "directional_alignment": float(cosine_rows.mean().cpu()),
        "revision_win_rate": float(revision_win.mean().cpu()),
    }
    by_rubric: dict[str, list[int]] = defaultdict(list)
    for row_index, record in enumerate(records):
        by_rubric[record.rubric].append(row_index)
    for rubric, indices in by_rubric.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        metrics[f"rubric/{rubric}/mse"] = float(mse_rows[index_tensor].mean().cpu())
        metrics[f"rubric/{rubric}/directional_alignment"] = float(
            cosine_rows[index_tensor].mean().cpu()
        )
    return metrics


def render_surrogate_image(
    record: PromptRecord,
    prediction: Sequence[float],
    target: Sequence[float],
    title: str,
    size: int = 384,
) -> Image.Image:
    """Render a compact visual diagnostic for the tiny image-main substrate."""

    image = Image.new("RGB", (size, size), color=(248, 248, 244))
    draw = ImageDraw.Draw(image)
    margin = 18
    bar_left = margin
    bar_right = size - margin
    bar_top = 116
    bar_height = 12
    max_width = bar_right - bar_left
    draw.rectangle((0, 0, size, 78), fill=(34, 45, 62))
    draw.text((margin, 12), title[:44], fill=(255, 255, 255))
    draw.text((margin, 34), record.prompt_id, fill=(222, 231, 239))
    draw.text((margin, 56), record.rubric[:36], fill=(180, 210, 255))
    draw.text((margin, 86), record.failure_key[:48], fill=(20, 28, 38))
    for index, key in enumerate(FAILURE_KEYS):
        y = bar_top + index * 28
        target_value = float(target[index])
        pred_value = max(0.0, min(1.0, float(prediction[index])))
        draw.text((bar_left, y - 12), key.replace("_", " ")[:34], fill=(30, 35, 42))
        draw.rectangle((bar_left, y, bar_right, y + bar_height), fill=(224, 226, 230))
        draw.rectangle(
            (bar_left, y, bar_left + int(max_width * target_value), y + bar_height),
            fill=(40, 167, 69),
        )
        draw.rectangle(
            (
                bar_left,
                y + bar_height + 3,
                bar_left + int(max_width * pred_value),
                y + 2 * bar_height + 3,
            ),
            fill=(47, 111, 237),
        )
    draw.text(
        (margin, size - 34), "green: target   blue: prediction", fill=(55, 65, 81)
    )
    return image


def wandb_surrogate_images(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    limit: int = 8,
) -> list[wandb.Image]:
    """Save surrogate diagnostic images and return W&B image objects."""

    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[wandb.Image] = []
    for index, row in enumerate(rows[:limit]):
        record = PromptRecord(
            prompt_id=str(row["prompt_id"]),
            prompt=str(row["prompt"]),
            prompt_hash=str(row["prompt_hash"]),
            source=str(row.get("source", "eval_row")),
            category=str(row["rubric"]),
            split=str(row["split"]),
            used_for_critique_collection=False,
            used_for_reward_model_training=False,
            used_for_hyperparameter_tuning=False,
            official_benchmark_prompt=False,
            used_for_human_eval=False,
            rubric=str(row["rubric"]),
            failure_key=str(row["failure_key"]),
            failure_text=str(row.get("critique", "")),
        )
        image = render_surrogate_image(
            record,
            prediction=row["prediction"],
            target=row["target"],
            title=f"{prefix}: {row['method']} / {row['variant']}",
        )
        image_path = output_dir / f"{prefix}_{index:03d}.jpg"
        image.save(image_path)
        caption = (
            f"{row['method']} | {row['variant']} | {row['rubric']} | "
            f"{str(row['prompt'])[:160]}"
        )
        images.append(wandb.Image(str(image_path), caption=caption))
    return images


def log_training_images(
    model: TinyConditionedFlow,
    method: str,
    batch: dict[str, Any],
    device: torch.device,
    step: int,
) -> None:
    """Log native DiffusionNFT-style training images to W&B."""

    records = batch["records"]
    prompt = batch["prompt_features"].to(device)
    target = batch["target"].detach().cpu()
    if method == "critique_full":
        condition = batch["fix_condition"].to(device)
    elif method == "scalar_multi_reward":
        condition = batch["scalar_condition"].to(device)
    else:
        condition = batch["base_condition"].to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(prompt, condition).detach().cpu()
    model.train()
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "prompt": record.prompt,
                "prompt_hash": record.prompt_hash,
                "split": record.split,
                "method": method,
                "variant": "train",
                "rubric": record.rubric,
                "failure_key": record.failure_key,
                "critique": record.failure_text,
                "target": target[index].tolist(),
                "prediction": predictions[index].tolist(),
            }
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        images = wandb_surrogate_images(
            rows,
            output_dir=Path(tmpdir),
            prefix=f"{method}_step_{step:04d}",
            limit=8,
        )
        wandb.log({"images": images})


def evaluate_model(
    model: TinyConditionedFlow,
    records: Sequence[PromptRecord],
    method: str,
    device: torch.device,
    variant: str = "default",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a trained model on records with one condition variant."""

    model.eval()
    if method == "critique_full" and variant == "default":
        variant = "correct"
    dataset = ImageMainToyDataset(records)
    rows: list[dict[str, Any]] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for index, example in enumerate(dataset):
            record = example["record"]
            prompt = example["prompt_features"].unsqueeze(0).to(device)
            target = example["target"].unsqueeze(0).to(device)
            if method == "critique_full":
                condition_text, critique = condition_for_variant(
                    record, variant, records, index
                )
                condition = (
                    condition_feature_vector(condition_text).unsqueeze(0).to(device)
                )
            elif method == "scalar_multi_reward":
                critique = SCALAR_CONDITION
                condition_text = SCALAR_CONDITION
                condition = example["scalar_condition"].unsqueeze(0).to(device)
            elif method == "positive_only":
                critique = BASE_CONDITION
                condition_text = BASE_CONDITION
                condition = example["base_condition"].unsqueeze(0).to(device)
            elif method == "base":
                critique = BASE_CONDITION
                condition_text = BASE_CONDITION
                condition = example["base_condition"].unsqueeze(0).to(device)
            else:
                raise ValueError(f"unknown method: {method}")
            prediction = model(prompt, condition)
            predictions.append(prediction.squeeze(0).detach().cpu())
            targets.append(target.squeeze(0).detach().cpu())
            rows.append(
                {
                    "prompt_id": record.prompt_id,
                    "prompt": record.prompt,
                    "prompt_hash": record.prompt_hash,
                    "source": record.source,
                    "split": record.split,
                    "method": method,
                    "variant": variant,
                    "rubric": record.rubric,
                    "failure_key": record.failure_key,
                    "critique": critique,
                    "condition": condition_text,
                    "target": target.squeeze(0).detach().cpu().tolist(),
                    "prediction": prediction.squeeze(0).detach().cpu().tolist(),
                }
            )
    metrics = evaluate_outputs(torch.stack(predictions), torch.stack(targets), records)
    return metrics, rows


def prepare_wandb(cfg: ImageMainConfig, metadata: dict[str, Any]) -> Any | None:
    """Create a W&B run for project experiments."""

    if cfg.wandb_mode == "disabled":
        if not cfg.allow_disabled_wandb:
            raise RuntimeError("wandb disabled mode requires --allow-disabled-wandb")
        return None
    if cfg.wandb_mode != "online":
        raise RuntimeError("WANDB_MODE must be online for group-01 experiment runs")
    wandb_dir = Path(os.environ.get("WANDB_DIR", "outputs/wandb")).resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        mode="online",
        dir=str(wandb_dir),
        tags=["experiment-01", "image-main", cfg.phase, "diffusionnft-substrate"],
        config=metadata,
    )


def train_method(
    method: str,
    cfg: ImageMainConfig,
    train_records: Sequence[PromptRecord],
    eval_records: Sequence[PromptRecord],
    device: torch.device,
    wandb_run: Any | None,
) -> tuple[TinyConditionedFlow, dict[str, float]]:
    """Train one tiny same-substrate method and return evaluation metrics."""

    dataset = ImageMainToyDataset(train_records)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 17),
        collate_fn=collate_batch,
    )
    iterator = iter(loader)
    model = make_model(cfg.hidden_dim, device)
    if method == "base":
        metrics, _ = evaluate_model(model, eval_records, method, device)
        return model, {f"{method}/{key}": value for key, value in metrics.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    write_jsonl(
        cfg.log_path,
        {
            "event": "model_constructed",
            "method": method,
            "hidden_dim": cfg.hidden_dim,
            "device": str(device),
        },
    )
    for step in range(cfg.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if method == "critique_full":
            loss, loss_parts = cnft_loss(model, batch, device)
        elif method == "scalar_multi_reward":
            loss, loss_parts = scalar_loss(model, batch, device)
        elif method == "positive_only":
            loss, loss_parts = positive_only_loss(model, batch, device)
        else:
            raise ValueError(f"unknown method: {method}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {method} at step {step}")
        loss.backward()
        grad = gradient_norm(model.parameters())
        if grad <= 0.0:
            raise RuntimeError(f"zero gradient for {method} at step {step}")
        optimizer.step()
        payload = {
            "event": "train_step",
            "method": method,
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad,
            **loss_parts,
        }
        write_jsonl(cfg.log_path, payload)
        if wandb_run is not None:
            wandb_payload = {
                f"{method}/train_loss": float(loss.detach().cpu()),
                f"{method}/grad_norm": grad,
            }
            wandb_payload.update(
                {f"{method}/{key}": value for key, value in loss_parts.items()}
            )
            wandb.log(wandb_payload)
            if step in {0, cfg.max_steps - 1}:
                log_training_images(model, method, batch, device, step)

    metrics, _ = evaluate_model(model, eval_records, method, device)
    return model, {f"{method}/{key}": value for key, value in metrics.items()}


def run_initial_forward_backward_check(
    cfg: ImageMainConfig,
    train_records: Sequence[PromptRecord],
    device: torch.device,
) -> dict[str, float]:
    """Verify model construction, forward pass, loss, and gradient flow."""

    dataset = ImageMainToyDataset(train_records)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_batch)
    batch = next(iter(loader))
    model = make_model(cfg.hidden_dim, device)
    loss, loss_parts = cnft_loss(model, batch, device)
    loss.backward()
    grad = gradient_norm(model.parameters())
    if not torch.isfinite(loss):
        raise RuntimeError("initial CNFT loss is non-finite")
    if grad <= 0.0:
        raise RuntimeError("initial CNFT gradient norm must be positive")
    result = {
        "initial_loss": float(loss.detach().cpu()),
        "initial_grad_norm": grad,
        **loss_parts,
    }
    write_jsonl(cfg.log_path, {"event": "initial_forward_backward", **result})
    return result


def save_checkpoint(
    checkpoint_dir: Path,
    models: dict[str, TinyConditionedFlow],
    cfg: ImageMainConfig,
    metrics: dict[str, Any],
) -> Path:
    """Write a compact checkpoint for all tiny methods."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "tiny_image_main_methods.pt"
    torch.save(
        {
            "models": {name: model.state_dict() for name, model in models.items()},
            "config": serializable_config(cfg),
            "metrics": metrics,
            "failure_keys": FAILURE_KEYS,
        },
        checkpoint_path,
    )
    return checkpoint_path


def serializable_config(cfg: ImageMainConfig) -> dict[str, Any]:
    """Convert a config dataclass into a JSON-serializable dictionary."""

    payload = asdict(cfg)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Write flat metrics to CSV for quick inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                writer.writerow([key, value])


def write_rollout_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write evaluation rows to a rollout JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            payload["artifact_path"] = str(path)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_critique_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write critique assignments to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "prompt_id": row["prompt_id"],
                "prompt_hash": row["prompt_hash"],
                "split": row["split"],
                "method": row["method"],
                "variant": row["variant"],
                "rubric": row["rubric"],
                "failure_key": row["failure_key"],
                "critique": row["critique"],
                "condition": row["condition"],
                "scalar_free": not any(
                    token in row["critique"].lower()
                    for token in ("score", "rank", "pass")
                ),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def plot_alignment(path: Path, metrics: dict[str, Any]) -> None:
    """Create a compact semantic-control alignment bar chart."""

    keys = [
        "base/directional_alignment",
        "scalar_multi_reward/directional_alignment",
        "positive_only/directional_alignment",
        "critique_full/correct/directional_alignment",
        "critique_full/paraphrase/directional_alignment",
        "critique_full/generic/directional_alignment",
        "critique_full/random_shuffled/directional_alignment",
    ]
    labels = [
        "base",
        "scalar",
        "positive",
        "correct",
        "paraphrase",
        "generic",
        "shuffled",
    ]
    values = [float(metrics.get(key, 0.0)) for key in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    colors = [
        "#5b667a",
        "#3b82f6",
        "#14b8a6",
        "#22c55e",
        "#84cc16",
        "#f97316",
        "#ef4444",
    ]
    axis.bar(labels, values, color=colors)
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("Directional alignment")
    axis.set_title("Experiment 01 tiny semantic-control readiness")
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix flat metric names."""

    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def run_image_main(cfg: ImageMainConfig) -> dict[str, Any]:
    """Run the group-01 manifest, smoke, training, evaluation, and artifact path."""

    set_seed(cfg.seed)
    device = select_device(cfg.device)
    run_artifact_dir = cfg.artifact_root / cfg.run_name
    checkpoint_dir = cfg.checkpoint_root / cfg.run_name
    metrics_dir = cfg.artifact_root / "metrics" / cfg.run_name
    rollout_dir = cfg.artifact_root / "rollouts" / cfg.run_name
    critique_dir = cfg.artifact_root / "critiques" / cfg.run_name
    eval_sample_dir = cfg.artifact_root / "eval_samples" / cfg.run_name
    for directory in (
        run_artifact_dir,
        checkpoint_dir,
        metrics_dir,
        rollout_dir,
        critique_dir,
        eval_sample_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if cfg.log_path.exists():
        cfg.log_path.unlink()

    manifest_paths = create_prompt_manifests(cfg)
    manifests = load_prompt_manifests(manifest_paths)
    split_counts = {split: len(records) for split, records in manifests.items()}
    write_jsonl(
        cfg.log_path,
        {
            "event": "manifests_written",
            "paths": {split: str(path) for split, path in manifest_paths.items()},
            "split_counts": split_counts,
        },
    )
    train_records = manifests["train"]
    eval_records = manifests["val"] + manifests["test_id"] + manifests["test_ood"]
    dataset_metadata = {
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "prompt_feature_dim": int(prompt_feature_vector(train_records[0]).numel()),
        "condition_feature_dim": int(condition_feature_vector(BASE_CONDITION).numel()),
        "target_dim": len(FAILURE_KEYS),
    }
    write_jsonl(cfg.log_path, {"event": "dataset_loaded", **dataset_metadata})

    initial_check = run_initial_forward_backward_check(cfg, train_records, device)
    metadata = {
        "phase": cfg.phase,
        "run_name": cfg.run_name,
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "manifest_paths": {split: str(path) for split, path in manifest_paths.items()},
        "artifact_dir": str(run_artifact_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(cfg.log_path),
        "dataset": dataset_metadata,
        "config": serializable_config(cfg),
        "initial_check": initial_check,
    }
    (run_artifact_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    wandb_run = prepare_wandb(cfg, metadata)

    methods = ("base", "scalar_multi_reward", "positive_only", "critique_full")
    models: dict[str, TinyConditionedFlow] = {}
    metrics: dict[str, Any] = {
        "phase": cfg.phase,
        "run_name": cfg.run_name,
        "split_counts": split_counts,
        **initial_check,
    }
    for method in methods:
        model, method_metrics = train_method(
            method, cfg, train_records, eval_records, device, wandb_run
        )
        models[method] = model
        metrics.update(method_metrics)
        write_jsonl(
            cfg.log_path, {"event": "evaluation", "method": method, **method_metrics}
        )

    rollout_rows: list[dict[str, Any]] = []
    critique_rows: list[dict[str, Any]] = []
    semantic_variants = (
        "correct",
        "paraphrase",
        "same_rubric_wrong_instance",
        "prompt_only_revision",
        "generic",
        "random_shuffled",
        "wrong_rubric",
    )
    critique_model = models["critique_full"]
    for variant in semantic_variants:
        variant_metrics, variant_rows = evaluate_model(
            critique_model,
            eval_records,
            "critique_full",
            device,
            variant=variant,
        )
        prefixed = flatten_metrics(f"critique_full/{variant}", variant_metrics)
        metrics.update(prefixed)
        rollout_rows.extend(variant_rows)
        critique_rows.extend(variant_rows)
        write_jsonl(
            cfg.log_path,
            {"event": "semantic_control_eval", "variant": variant, **prefixed},
        )

    correct_alignment = float(metrics["critique_full/correct/directional_alignment"])
    generic_alignment = float(metrics["critique_full/generic/directional_alignment"])
    shuffled_alignment = float(
        metrics["critique_full/random_shuffled/directional_alignment"]
    )
    scalar_alignment = float(metrics["scalar_multi_reward/directional_alignment"])
    metrics["semantic_gap/correct_minus_generic_alignment"] = (
        correct_alignment - generic_alignment
    )
    metrics["semantic_gap/correct_minus_shuffled_alignment"] = (
        correct_alignment - shuffled_alignment
    )
    metrics["main_anchor/correct_minus_scalar_alignment"] = (
        correct_alignment - scalar_alignment
    )
    metrics["teacher_gate/correct_revision_win_rate"] = float(
        metrics["critique_full/correct/revision_win_rate"]
    )

    checkpoint_path = save_checkpoint(checkpoint_dir, models, cfg, metrics)
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["metrics_json"] = str(metrics_dir / "metrics.json")
    metrics["metrics_csv"] = str(metrics_dir / "metrics.csv")

    rollout_path = rollout_dir / "semantic_control_rollouts.jsonl"
    critique_path = critique_dir / "critique_assignments.jsonl"
    write_rollout_rows(rollout_path, rollout_rows)
    write_critique_rows(critique_path, critique_rows)
    metrics["rollout_jsonl"] = str(rollout_path)
    metrics["critique_jsonl"] = str(critique_path)

    eval_image_rows = [
        row
        for row in rollout_rows
        if row["variant"] in {"correct", "generic", "random_shuffled", "wrong_rubric"}
    ]
    eval_image_dir = eval_sample_dir / "semantic_control_images"
    eval_wandb_images = wandb_surrogate_images(
        eval_image_rows,
        output_dir=eval_image_dir,
        prefix="eval",
        limit=16,
    )
    metrics["eval_image_dir"] = str(eval_image_dir)

    plot_path = eval_sample_dir / "semantic_control_alignment.png"
    plot_alignment(plot_path, metrics)
    metrics["alignment_plot"] = str(plot_path)

    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_metrics_csv(metrics_dir / "metrics.csv", metrics)
    write_jsonl(
        cfg.log_path, {"event": "checkpoint_written", "path": str(checkpoint_path)}
    )
    write_jsonl(
        cfg.log_path,
        {
            "event": "artifacts_written",
            "metrics_json": metrics["metrics_json"],
            "rollout_jsonl": str(rollout_path),
            "critique_jsonl": str(critique_path),
            "alignment_plot": str(plot_path),
        },
    )

    if wandb_run is not None:
        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        wandb.log(numeric_metrics)
        wandb.log({"semantic_control_alignment": wandb.Image(str(plot_path))})
        wandb.log({"eval_images": eval_wandb_images})
        wandb_run.summary.update(metrics)
        wandb_run.finish()
    return metrics


def phase_default_counts(phase: str) -> dict[str, int]:
    """Return default sizes for a phase."""

    if phase == "smoke":
        return {
            "train_count": 16,
            "val_count": 8,
            "test_count": 8,
            "ood_count": 6,
            "max_steps": 4,
            "batch_size": 4,
        }
    return {
        "train_count": 96,
        "val_count": 24,
        "test_count": 24,
        "ood_count": 18,
        "max_steps": 64,
        "batch_size": 12,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "debug-real"], default="smoke")
    parser.add_argument(
        "--run-name", default=os.environ.get("WANDB_RUN_NAME", "01-image-main-smoke")
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/artifacts/image_main/prompts")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("data/artifacts/image_main")
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("data/checkpoints/image_main")
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/logs/image_main/01_image_main.jsonl"),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--ood-count", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "world-reward-models"),
    )
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--allow-disabled-wandb", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ImageMainConfig:
    """Build a config from parsed arguments."""

    defaults = phase_default_counts(args.phase)
    return ImageMainConfig(
        phase=args.phase,
        run_name=args.run_name,
        manifest_root=args.manifest_root,
        artifact_root=args.artifact_root,
        checkpoint_root=args.checkpoint_root,
        log_path=args.log_path,
        dataset_root=args.dataset_root,
        seed=args.seed,
        train_count=(
            args.train_count
            if args.train_count is not None
            else defaults["train_count"]
        ),
        val_count=(
            args.val_count if args.val_count is not None else defaults["val_count"]
        ),
        test_count=(
            args.test_count if args.test_count is not None else defaults["test_count"]
        ),
        ood_count=(
            args.ood_count if args.ood_count is not None else defaults["ood_count"]
        ),
        max_steps=(
            args.max_steps if args.max_steps is not None else defaults["max_steps"]
        ),
        batch_size=(
            args.batch_size if args.batch_size is not None else defaults["batch_size"]
        ),
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        allow_disabled_wandb=args.allow_disabled_wandb,
    )


def main() -> None:
    """Run Experiment 01 image-main readiness."""

    cfg = config_from_args(parse_args())
    metrics = run_image_main(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
