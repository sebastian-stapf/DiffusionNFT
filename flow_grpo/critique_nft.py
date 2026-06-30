import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

Z0_CONDITION = "high-quality, faithful, artifact-free generation."
GENERIC_CRITIQUE = "improve image quality and prompt faithfulness"

FAILURE_TEXT = {
    "alignment_missing_object": "a required object is missing or replaced",
    "alignment_spatial_reversal": "the spatial relationship is reversed or misplaced",
    "ocr_misspelled_text": "the requested text is misspelled",
    "ocr_unreadable_text": "the text is warped or unreadable",
    "quality_blur_noise": "the image is blurry or noisy",
    "quality_blocky_texture": "the texture is blocky or visibly corrupted",
    "aesthetic_poor_lighting": "the lighting and contrast are poor",
    "aesthetic_visual_clutter": "the composition is cluttered and distracting",
}

FAILURE_PARAPHRASE = {
    "alignment_missing_object": "an important requested item is absent",
    "alignment_spatial_reversal": "the left-right or above-below placement is wrong",
    "ocr_misspelled_text": "the rendered word has incorrect letters",
    "ocr_unreadable_text": "the lettering cannot be read clearly",
    "quality_blur_noise": "the image contains blur, grain, or fuzzy details",
    "quality_blocky_texture": "surfaces show damaged or block-like artifacts",
    "aesthetic_poor_lighting": "the scene has weak contrast and unbalanced tones",
    "aesthetic_visual_clutter": "too many distracting elements compete in the frame",
}

FAILURE_RUBRIC = {
    "alignment_missing_object": "alignment",
    "alignment_spatial_reversal": "count_color_spatial",
    "ocr_misspelled_text": "ocr",
    "ocr_unreadable_text": "ocr",
    "quality_blur_noise": "piqe_like_quality",
    "quality_blocky_texture": "piqe_like_quality",
    "aesthetic_poor_lighting": "aesthetic",
    "aesthetic_visual_clutter": "human_preference",
}

DATASET_FAILURE_KEYS = {
    "geneval": (
        "alignment_missing_object",
        "alignment_spatial_reversal",
    ),
    "ocr": (
        "ocr_misspelled_text",
        "ocr_unreadable_text",
    ),
    "pickscore": (
        "quality_blur_noise",
        "quality_blocky_texture",
        "aesthetic_poor_lighting",
        "aesthetic_visual_clutter",
    ),
}

SEMANTIC_VARIANTS = (
    "correct",
    "paraphrase",
    "same_rubric_wrong_instance",
    "prompt_only_revision",
    "generic",
    "random_shuffled",
    "wrong_rubric",
    "no_delta",
    "no_negative_branch",
    "critique_without_revision_target",
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def z_bad(feedback: str) -> str:
    return f"A low-quality generation exhibiting these visible failures: {feedback}."


def z_fix(feedback: str) -> str:
    return f"A corrected generation that addresses these failures: {feedback}."


def conditioned_prompt(prompt: str, condition: str) -> str:
    return f"{prompt}\nCondition: {condition}"


def dataset_name_from_config(config: Any) -> str:
    dataset = str(getattr(config, "dataset", "")).rstrip("/")
    name = os.path.basename(dataset)
    if name in DATASET_FAILURE_KEYS:
        return name
    if getattr(config, "prompt_fn", "") == "geneval":
        return "geneval"
    return "pickscore"


def _geneval_failure_key(metadata: dict[str, Any], index: int) -> str:
    tag = str(metadata.get("tag", ""))
    if tag == "position":
        return "alignment_spatial_reversal"
    if tag in {"counting", "colors", "color_attr", "single_object", "two_object"}:
        return "alignment_missing_object"
    keys = DATASET_FAILURE_KEYS["geneval"]
    return keys[index % len(keys)]


def failure_key_for_prompt(dataset: str, prompt: str, metadata: dict[str, Any], index: int) -> str:
    del prompt
    if dataset == "geneval":
        return _geneval_failure_key(metadata, index)
    keys = DATASET_FAILURE_KEYS.get(dataset, DATASET_FAILURE_KEYS["pickscore"])
    return keys[index % len(keys)]


def build_critique_records(
    prompts: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    dataset: str,
    start_index: int = 0,
    split: str = "train",
) -> list[dict[str, Any]]:
    records = []
    for local_index, prompt in enumerate(prompts):
        global_index = start_index + local_index
        metadata = metadatas[local_index] if local_index < len(metadatas) else {}
        failure_key = failure_key_for_prompt(dataset, prompt, metadata, global_index)
        failure_text = FAILURE_TEXT[failure_key]
        records.append(
            {
                "prompt_id": f"{split}-{global_index:08d}",
                "prompt": prompt,
                "prompt_hash": stable_hash(prompt),
                "metadata": metadata,
                "split": split,
                "dataset": dataset,
                "failure_key": failure_key,
                "failure_text": failure_text,
                "rubric": FAILURE_RUBRIC[failure_key],
                "z_0": Z0_CONDITION,
                "z_bad": z_bad(failure_text),
                "z_fix": z_fix(failure_text),
            }
        )
    return records


def critique_for_variant(
    record: dict[str, Any],
    records: Sequence[dict[str, Any]],
    index: int,
    variant: str,
) -> str:
    if variant in {
        "correct",
        "no_delta",
        "no_negative_branch",
        "critique_without_revision_target",
    }:
        return str(record["failure_text"])
    if variant == "paraphrase":
        return FAILURE_PARAPHRASE[str(record["failure_key"])]
    if variant in {"prompt_only_revision", "generic"}:
        return GENERIC_CRITIQUE
    if variant == "same_rubric_wrong_instance":
        candidates = [
            key
            for key, rubric in FAILURE_RUBRIC.items()
            if rubric == record["rubric"] and key != record["failure_key"]
        ]
        if not candidates:
            candidates = [key for key in FAILURE_TEXT if key != record["failure_key"]]
        return FAILURE_TEXT[candidates[index % len(candidates)]]
    if variant == "random_shuffled":
        if not records:
            return GENERIC_CRITIQUE
        other = records[(index * 7 + 3) % len(records)]
        if other["failure_key"] == record["failure_key"] and len(records) > 1:
            other = records[(index * 7 + 4) % len(records)]
        return str(other["failure_text"])
    if variant == "wrong_rubric":
        candidates = [
            key
            for key, rubric in FAILURE_RUBRIC.items()
            if rubric != record["rubric"] and key != record["failure_key"]
        ]
        return FAILURE_TEXT[candidates[index % len(candidates)]]
    raise ValueError(f"unknown semantic-control variant: {variant}")


def prompts_for_role(
    records: Sequence[dict[str, Any]],
    role: str,
    variant: str = "correct",
    all_records: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    source_records = all_records or records
    prompts = []
    for index, record in enumerate(records):
        if role == "z0":
            condition = Z0_CONDITION
        elif role == "bad":
            critique = critique_for_variant(record, source_records, index, variant)
            condition = z_bad(critique)
        elif role == "fix":
            critique = critique_for_variant(record, source_records, index, variant)
            condition = z_fix(critique)
        else:
            raise ValueError(f"unknown role: {role}")
        prompts.append(conditioned_prompt(str(record["prompt"]), condition))
    return prompts


def records_to_manifest_rows(
    records: Sequence[dict[str, Any]], used_for_training: bool
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
                "prompt_hash": record["prompt_hash"],
                "source": f"diffusionnft_{record['dataset']}",
                "category": record["rubric"],
                "split": record["split"],
                "used_for_critique_collection": used_for_training,
                "used_for_reward_model_training": False,
                "used_for_hyperparameter_tuning": False,
                "official_benchmark_prompt": record["split"] != "train",
                "used_for_human_eval": record["split"] != "train",
                "rubric": record["rubric"],
                "failure_key": record["failure_key"],
                "failure_text": record["failure_text"],
                "z_0": record["z_0"],
                "z_bad": record["z_bad"],
                "z_fix": record["z_fix"],
            }
        )
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def image_tensor_to_pil(image) -> Image.Image:
    array = (
        (image.float().cpu().numpy().transpose(1, 2, 0) * 255)
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(array)


def write_image_grid_rows(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    draft_images,
    revision_images,
    draft_rewards: Sequence[float],
    revision_rewards: Sequence[float],
    limit: int,
    prefix: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    count = min(limit, len(records), len(draft_images), len(revision_images))
    for index in range(count):
        draft_path = output_dir / f"{prefix}_{index:03d}_draft.jpg"
        revision_path = output_dir / f"{prefix}_{index:03d}_revision.jpg"
        image_tensor_to_pil(draft_images[index]).save(draft_path)
        image_tensor_to_pil(revision_images[index]).save(revision_path)
        row = {
            "prompt_id": records[index]["prompt_id"],
            "prompt": records[index]["prompt"],
            "prompt_hash": records[index]["prompt_hash"],
            "failure_key": records[index]["failure_key"],
            "critique": records[index]["failure_text"],
            "draft_image": str(draft_path),
            "revision_image": str(revision_path),
            "draft_reward": float(draft_rewards[index]),
            "revision_reward": float(revision_rewards[index]),
            "revision_wins": float(revision_rewards[index]) > float(draft_rewards[index]),
        }
        rows.append(row)
    return rows


def scalar_weight_grid() -> list[dict[str, float]]:
    return [
        {"geneval": 1.0, "ocr": 0.0, "pickscore": 0.0},
        {"geneval": 0.0, "ocr": 1.0, "pickscore": 0.0},
        {"geneval": 0.0, "ocr": 0.0, "pickscore": 1.0},
        {"geneval": 0.5, "ocr": 0.5, "pickscore": 0.0},
        {"geneval": 0.5, "ocr": 0.0, "pickscore": 0.5},
        {"geneval": 0.0, "ocr": 0.5, "pickscore": 0.5},
        {"geneval": 1.0 / 3.0, "ocr": 1.0 / 3.0, "pickscore": 1.0 / 3.0},
    ]


def write_run_scaffolds(
    artifact_root: Path, run_name: str, config_payload: dict[str, Any]
) -> dict[str, str]:
    run_root = artifact_root / "native_critique_nft" / run_name
    paths = {
        "run_root": str(run_root),
        "stage_a_warm_jsonl": str(run_root / "stage_a_warm_pairs.jsonl"),
        "stage_b_teacher_gate_jsonl": str(run_root / "stage_b_teacher_gate.jsonl"),
        "stage_c_rollout_jsonl": str(run_root / "stage_c_on_policy_rollouts.jsonl"),
        "semantic_control_jsonl": str(run_root / "semantic_control_eval.jsonl"),
        "scalar_sweep_json": str(run_root / "scalar_weight_sweep.json"),
        "human_eval_manifest_jsonl": str(run_root / "human_eval_pairs.jsonl"),
        "summary_json": str(run_root / "summary.json"),
    }
    write_json(
        Path(paths["scalar_sweep_json"]),
        {
            "status": "ready_pending_baseline_runs",
            "reason": "Scalar baseline runs are intentionally skipped in this pass and will be run later.",
            "grid": scalar_weight_grid(),
            "config": config_payload,
        },
    )
    write_json(
        Path(paths["summary_json"]),
        {
            "status": "running",
            "artifacts": paths,
            "config": config_payload,
        },
    )
    return paths


def write_human_eval_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payloads = []
    for row in rows:
        payloads.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "critique_nft_image": row["revision_image"],
                "scalar_baseline_image": None,
                "status": "pending_scalar_baseline",
                "questions": [
                    "Which image better follows the prompt?",
                    "Which image has fewer visible artifacts?",
                    "Which image is better overall?",
                ],
            }
        )
    append_jsonl(path, payloads)
